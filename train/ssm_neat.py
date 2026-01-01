from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from train.train_ssm import (  # noqa: E402
    GeoPlantDataset,
    build_cb_alpha,
    build_class_priors,
    build_labels,
    build_pos_weight,
    build_species_vocab,
    cb_focal_loss,
    load_env_features,
    load_time_series,
    logit_adjusted_loss,
    make_weighted_sampler,
    run_epoch,
    seed_everything,
    split_ids,
)
from src.models.ssm.ssm import StateSpaceModel  # noqa: E402


@dataclass
class DataBundle:
    train_ds: GeoPlantDataset
    val_ds: GeoPlantDataset
    train_labels: np.ndarray
    env_dim: int
    ts_channels: int


@dataclass
class Candidate:
    params: Dict[str, float | int | str | bool]
    fitness: float = float("-inf")
    history: List[Tuple[int, float]] = None

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolutionary hyperparameter search for SSM.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--species-count", type=int, default=342)
    parser.add_argument("--species-vocab", type=Path, default=Path("data/species_vocab.json"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-train", type=int, default=1024, help="Cap train samples per candidate for speed.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per candidate evaluation.")
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--mutation-prob", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"], default="offline")
    parser.add_argument("--project", type=str, default="geo-plant-ssm-neat")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--no-env", action="store_true")
    parser.add_argument("--no-ts", action="store_true")
    parser.add_argument("--no-img", action="store_true")
    return parser.parse_args()


def sample_hyperparams() -> Dict[str, float | int | str | bool]:
    return {
        "state_dim": random.choice([192, 256, 320, 384]),
        "lambda_reg": 10 ** random.uniform(math.log10(0.01), math.log10(0.1)),
        "lr": 10 ** random.uniform(math.log10(3e-5), math.log10(3e-4)),
        "loss_type": random.choice(["bce", "cb_focal", "logit_adj"]),
        "cb_beta": random.uniform(0.95, 0.9999),
        "focal_gamma": random.uniform(1.0, 3.0),
        "logit_tau": random.uniform(0.5, 2.0),
        "use_sampler": random.choice([True, False]),
        "dropout": random.choice([0.05, 0.1, 0.2]),
        "unfreeze_img": False,
    }


def mutate(params: Dict[str, float | int | str | bool], mutation_prob: float) -> Dict[str, float | int | str | bool]:
    new_params = params.copy()
    for key in params:
        if random.random() > mutation_prob:
            continue
        if key == "state_dim":
            new_params[key] = random.choice([192, 256, 320, 384, 448])
        elif key == "lambda_reg":
            new_params[key] = 10 ** random.uniform(math.log10(0.01), math.log10(0.1))
        elif key == "lr":
            new_params[key] = 10 ** random.uniform(math.log10(3e-5), math.log10(5e-4))
        elif key == "loss_type":
            new_params[key] = random.choice(["bce", "cb_focal", "logit_adj"])
        elif key == "cb_beta":
            new_params[key] = random.uniform(0.9, 0.9999)
        elif key == "focal_gamma":
            new_params[key] = random.uniform(1.0, 4.0)
        elif key == "logit_tau":
            new_params[key] = random.uniform(0.3, 2.5)
        elif key == "use_sampler":
            new_params[key] = not bool(params[key])
        elif key == "dropout":
            new_params[key] = random.choice([0.05, 0.1, 0.15, 0.2, 0.25])
        elif key == "unfreeze_img":
            new_params[key] = not bool(params[key])
    return new_params


def prepare_data(args: argparse.Namespace, use_img: bool) -> DataBundle:
    metadata_train = args.data_root / "PA_metadata_train.csv"
    env_path = args.data_root / "EnvironmentalValues/Climate/Average 1981-2010/PA-train-bioclimatic.csv"
    ts_root = args.data_root / "SateliteTimeSeries-Landsat/values"
    rgb_root = args.data_root / "SatelitePatches/PA-train-RGB"

    vocab, _ = build_species_vocab(metadata_train, args.species_count, args.species_vocab)
    labels, label_sids = build_labels(metadata_train, vocab)

    env = load_env_features(env_path)
    ts, ts_sids = load_time_series(ts_root, "train")

    common_ids = set(label_sids) & set(env.index.tolist()) & set(ts_sids)
    if use_img:
        from train.train_ssm import filter_ids_with_images

        common_ids = filter_ids_with_images(sorted(common_ids), rgb_root)
    else:
        common_ids = sorted(common_ids)

    train_ids, val_ids = split_ids(common_ids, args.val_ratio, seed=args.seed)
    if args.max_train:
        train_ids = list(train_ids)[: args.max_train]

    label_index = {sid: idx for idx, sid in enumerate(label_sids)}
    train_label_rows = [label_index[sid] for sid in train_ids]
    val_label_rows = [label_index[sid] for sid in val_ids]

    train_labels = labels[train_label_rows]
    val_labels = labels[val_label_rows]

    train_ds = GeoPlantDataset(
        survey_ids=train_ids,
        labels=train_labels,
        env=env,
        ts=ts,
        ts_sids=ts_sids,
        rgb_root=rgb_root,
        image_size=224,
        use_images=use_img,
    )
    val_ds = GeoPlantDataset(
        survey_ids=val_ids,
        labels=val_labels,
        env=env,
        ts=ts,
        ts_sids=ts_sids,
        rgb_root=rgb_root,
        image_size=224,
        use_images=use_img,
    )

    return DataBundle(
        train_ds=train_ds,
        val_ds=val_ds,
        train_labels=train_labels,
        env_dim=train_ds.env.shape[1],
        ts_channels=train_ds.ts.shape[2],
    )


def make_loss_fn(params: Dict[str, float | int | str | bool], train_labels: np.ndarray, device: torch.device):
    pos_weight = build_pos_weight(train_labels, cap=50.0).to(device)
    class_priors = build_class_priors(train_labels)
    cb_alpha = build_cb_alpha(train_labels, beta=float(params["cb_beta"]))

    if params["loss_type"] == "bce":
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        def loss_fn(logits, labels):
            return criterion(logits, labels)

        return loss_fn

    if params["loss_type"] == "cb_focal":
        alpha = cb_alpha
        gamma = float(params["focal_gamma"])

        def loss_fn(logits, labels):
            return cb_focal_loss(logits, labels, alpha=alpha, gamma=gamma)

        return loss_fn

    logit_prior = torch.log(class_priors / (1 - class_priors))
    tau = float(params["logit_tau"])

    def loss_fn(logits, labels):
        return logit_adjusted_loss(logits, labels, logit_prior=logit_prior, tau=tau)

    return loss_fn


def make_loaders(
    data: DataBundle,
    batch_size: int,
    use_sampler: bool,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    sampler = make_weighted_sampler(data.train_labels) if use_sampler else None
    train_loader = DataLoader(
        data.train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        data.val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def evaluate_candidate(
    cand: Candidate,
    data: DataBundle,
    args: argparse.Namespace,
    device: torch.device,
    use_env: bool,
    use_ts: bool,
    use_img: bool,
    step: int,
) -> float:
    loss_fn = make_loss_fn(cand.params, data.train_labels, device)
    train_loader, val_loader = make_loaders(
        data=data,
        batch_size=args.batch_size,
        use_sampler=bool(cand.params["use_sampler"]),
        num_workers=args.num_workers,
    )

    model = StateSpaceModel(
        num_species=args.species_count,
        state_dim=int(cand.params["state_dim"]),
        env_dim=data.env_dim,
        ts_channels=data.ts_channels,
        img_freeze_backbone=not bool(cand.params["unfreeze_img"]),
        dropout=float(cand.params["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cand.params["lr"]))
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    last_val = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            lambda_reg=float(cand.params["lambda_reg"]),
            use_env=use_env,
            use_ts=use_ts,
            use_img=use_img,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            loss_fn,
            device,
            lambda_reg=float(cand.params["lambda_reg"]),
            use_env=use_env,
            use_ts=use_ts,
            use_img=use_img,
            optimizer=None,
            scaler=None,
        )
        last_val = val_metrics

        wandb.log(
            {
                "step": step,
                "cand/state_dim": cand.params["state_dim"],
                "cand/lambda_reg": cand.params["lambda_reg"],
                "cand/lr": cand.params["lr"],
                "cand/loss_type": cand.params["loss_type"],
                "cand/cb_beta": cand.params["cb_beta"],
                "cand/focal_gamma": cand.params["focal_gamma"],
                "cand/logit_tau": cand.params["logit_tau"],
                "cand/use_sampler": cand.params["use_sampler"],
                "cand/dropout": cand.params["dropout"],
                "cand/unfreeze_img": cand.params["unfreeze_img"],
                "train/loss": train_metrics.loss,
                "val/loss": val_metrics.loss,
                "val/macro_auc": val_metrics.macro_auc,
                "val/micro_auc": val_metrics.micro_auc,
                "val/macro_ap": val_metrics.macro_ap,
                "val/recall": val_metrics.recall,
            },
            step=step,
        )

    fitness = last_val.macro_auc if last_val is not None else float("-inf")
    cand.fitness = fitness
    cand.history.append((step, fitness))
    return fitness


def main() -> None:
    load_dotenv()
    args = parse_args()
    seed_everything(args.seed)

    use_env = not args.no_env
    use_ts = not args.no_ts
    use_img = not args.no_img

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = prepare_data(args, use_img=use_img)

    wandb_mode = args.wandb_mode
    if wandb_mode == "disabled":
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project=args.project,
            entity=args.entity,
            mode=wandb_mode,
            config=vars(args),
        )

    population: List[Candidate] = [Candidate(sample_hyperparams()) for _ in range(args.population)]
    global_step = 0

    for gen in range(args.generations):
        print(f"\nGeneration {gen + 1}/{args.generations}")
        for cand in tqdm(population, desc="candidates"):
            fitness = evaluate_candidate(
                cand,
                data=data,
                args=args,
                device=device,
                use_env=use_env,
                use_ts=use_ts,
                use_img=use_img,
                step=global_step,
            )
            global_step += 1
            print(f"  cand fitness (macro_auc)={fitness:.4f} params={cand.params}")

        population = sorted(population, key=lambda c: c.fitness, reverse=True)
        elites = population[: args.top_k]
        print(f"  Best fitness this gen: {elites[0].fitness:.4f}")

        children: List[Candidate] = []
        while len(children) + len(elites) < args.population:
            parent = random.choice(elites)
            child_params = mutate(parent.params, args.mutation_prob)
            children.append(Candidate(child_params))

        population = elites + children

    best = max(population, key=lambda c: c.fitness)
    print("\nBest candidate found:")
    print(best.params)
    print(f"macro_auc={best.fitness:.4f}")
    wandb.summary["best_macro_auc"] = best.fitness
    for k, v in best.params.items():
        wandb.summary[f"best_{k}"] = v
    wandb.finish()


if __name__ == "__main__":
    main()
