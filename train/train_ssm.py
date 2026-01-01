from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm.auto import tqdm
from scipy.special import expit

# Ensure local src/ is importable when running as a script
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.ssm.ssm import StateSpaceModel  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_species_vocab(
    metadata_path: Path,
    num_species: int,
    vocab_path: Optional[Path] = None,
) -> Tuple[Dict[int, int], pd.Series]:
    df = pd.read_csv(metadata_path)
    df["speciesId"] = df["speciesId"].astype(int)
    freq = df["speciesId"].value_counts()

    if vocab_path is not None and vocab_path.exists():
        vocab = {int(k): int(v) for k, v in json.loads(vocab_path.read_text()).items()}
    else:
        top_species = freq.index[:num_species]
        vocab = {int(sid): idx for idx, sid in enumerate(top_species)}
        if vocab_path is not None:
            vocab_path.write_text(json.dumps(vocab))

    return vocab, freq


def build_labels(
    metadata_path: Path, vocab: Dict[int, int]
) -> Tuple[np.ndarray, List[int]]:
    df = pd.read_csv(metadata_path)
    df["speciesId"] = df["speciesId"].astype(int)
    grouped = df.groupby("surveyId")["speciesId"].apply(list)

    survey_ids = sorted(grouped.index.tolist())
    labels = np.zeros((len(survey_ids), len(vocab)), dtype=np.float32)

    for row_idx, sid in enumerate(survey_ids):
        for sp in grouped.loc[sid]:
            if sp in vocab:
                labels[row_idx, vocab[sp]] = 1.0
    return labels, survey_ids


def load_env_features(env_path: Path) -> pd.DataFrame:
    env = pd.read_csv(env_path)
    env = env.set_index("surveyId").sort_index()
    return env


def load_time_series(ts_root: Path, split: str) -> Tuple[np.ndarray, List[int]]:
    band_names = ["blue", "green", "red", "nir", "swir1", "swir2"]
    band_arrays = []
    survey_ids: Optional[np.ndarray] = None
    for band in band_names:
        csv_path = ts_root / f"PA-{split}-landsat_time_series" / f"PA-{split}-landsat_time_series-{band}.csv"
        df = pd.read_csv(csv_path)
        df = df.sort_values("surveyId")
        if survey_ids is None:
            survey_ids = df["surveyId"].to_numpy(np.int64)
        cols = [c for c in df.columns if c != "surveyId"]
        arr = df[cols].to_numpy(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        band_arrays.append(arr)

    assert survey_ids is not None
    stacked = np.stack(band_arrays, axis=2)  # (N, T, C)
    return stacked, survey_ids.tolist()


def patch_path(root: Path, survey_id: int) -> Path:
    last2 = survey_id % 100
    prev2 = (survey_id // 100) % 100
    return root / str(last2) / str(prev2) / f"{survey_id}.jpeg"


def filter_ids_with_images(ids: Sequence[int], rgb_root: Path) -> List[int]:
    keep: List[int] = []
    missing = 0
    for sid in ids:
        if patch_path(rgb_root, sid).exists():
            keep.append(sid)
        else:
            missing += 1
    if missing:
        print(f"Skipping {missing} surveys without RGB patches.")
    return keep


class GeoPlantDataset(Dataset):
    def __init__(
        self,
        survey_ids: Sequence[int],
        labels: np.ndarray,
        env: pd.DataFrame,
        ts: np.ndarray,
        ts_sids: Sequence[int],
        rgb_root: Path,
        image_size: int = 224,
        use_images: bool = True,
    ):
        self.survey_ids = list(survey_ids)
        self.labels = labels
        self.env = env
        self.ts = ts
        self.ts_index = {sid: idx for idx, sid in enumerate(ts_sids)}
        self.rgb_root = rgb_root
        self.use_images = use_images
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.survey_ids)

    def __getitem__(self, idx: int):
        sid = self.survey_ids[idx]
        label = torch.from_numpy(self.labels[idx])

        env_vec = torch.tensor(
            self.env.loc[sid].to_numpy(np.float32), dtype=torch.float32
        )

        ts_idx = self.ts_index[sid]
        ts_tensor = torch.tensor(self.ts[ts_idx], dtype=torch.float32)

        if self.use_images:
            img_path = patch_path(self.rgb_root, sid)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found for surveyId={sid}: {img_path}")
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                img_tensor = self.transform(im)
        else:
            img_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)

        return env_vec, ts_tensor, img_tensor, label, sid


def split_ids(
    ids: Sequence[int], val_ratio: float, seed: int
) -> Tuple[List[int], List[int]]:
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    split = int(len(ids) * (1 - val_ratio))
    return ids[:split], ids[split:]


def build_pos_weight(labels: np.ndarray, cap: float = 50.0) -> torch.Tensor:
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    weight = neg / (pos + 1e-6)
    weight = np.clip(weight, 1.0, cap)
    return torch.tensor(weight, dtype=torch.float32)


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    pos_freq = labels.mean(axis=0)
    inv_freq = 1.0 / (pos_freq + 1e-6)
    sample_weights = []
    for row in labels:
        pos_idx = np.where(row == 1)[0]
        if len(pos_idx) > 0:
            sample_weights.append(inv_freq[pos_idx].mean())
        else:
            sample_weights.append(1.0)
    sample_weights = np.asarray(sample_weights, dtype=np.float32)
    sample_weights = sample_weights / sample_weights.mean()
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def build_class_priors(labels: np.ndarray, min_prior: float = 1e-4) -> torch.Tensor:
    prior = labels.mean(axis=0)
    prior = np.clip(prior, min_prior, 1 - min_prior)
    return torch.tensor(prior, dtype=torch.float32)


def build_cb_alpha(labels: np.ndarray, beta: float) -> torch.Tensor:
    counts = labels.sum(axis=0)
    effective_num = 1.0 - np.power(beta, counts)
    alpha = (1.0 - beta) / (effective_num + 1e-8)
    alpha = alpha / alpha.sum() * len(alpha)
    return torch.tensor(alpha, dtype=torch.float32)


def cb_focal_loss(
    logits: Tensor,
    targets: Tensor,
    alpha: Tensor,
    gamma: float,
) -> Tensor:
    targets = targets.float()
    alpha = alpha.to(logits.device)

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    focal_factor = (1 - p_t).pow(gamma)

    loss = focal_factor * bce * alpha
    return loss.mean()


def logit_adjusted_loss(
    logits: Tensor,
    targets: Tensor,
    logit_prior: Tensor,
    tau: float,
) -> Tensor:
    adj_logits = logits + tau * logit_prior.to(logits.device)
    return F.binary_cross_entropy_with_logits(adj_logits, targets.float())


@dataclass
class EpochMetrics:
    loss: float
    primary: float
    reg: float
    macro_auc: float
    micro_auc: float
    macro_ap: float
    recall: float


def compute_metrics(y_true: np.ndarray, logits: np.ndarray) -> Tuple[float, float, float, float]:
    logits = np.clip(logits, -30.0, 30.0)
    probs = expit(logits)
    macro_auc_list = []
    macro_ap_list = []
    for k in range(probs.shape[1]):
        y_k = y_true[:, k]
        if y_k.max() == 0 or y_k.min() == 1:
            continue
        macro_auc_list.append(roc_auc_score(y_k, probs[:, k]))
        macro_ap_list.append(average_precision_score(y_k, probs[:, k]))

    macro_auc = float(np.mean(macro_auc_list)) if macro_auc_list else float("nan")
    macro_ap = float(np.mean(macro_ap_list)) if macro_ap_list else float("nan")

    try:
        micro_auc = float(roc_auc_score(y_true, probs, average="micro"))
    except ValueError:
        micro_auc = float("nan")

    preds = (probs >= 0.5).astype(np.float32)
    recall = float((preds[y_true == 1] == 1).mean()) if (y_true == 1).any() else float("nan")

    return macro_auc, micro_auc, macro_ap, recall


def run_epoch(
    model: StateSpaceModel,
    loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    device: torch.device,
    lambda_reg: float,
    use_env: bool,
    use_ts: bool,
    use_img: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> EpochMetrics:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = total_primary = total_reg = 0.0
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for batch in tqdm(loader, desc="train" if is_train else "eval", leave=False):
        env, ts, img, labels, _ = batch
        env = env.to(device)
        ts = ts.to(device)
        img = img.to(device)
        labels = labels.to(device)

        with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda" and scaler is not None):
            outputs = model(
                env=env if use_env else None,
                ts=ts if use_ts else None,
                img=img if use_img else None,
            )
            logits = outputs.logits[-1]
            primary_loss = sum(loss_fn(logit, labels) for logit in outputs.logits) / len(outputs.logits)
            loss = primary_loss + lambda_reg * outputs.reg_loss

        if is_train:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach().cpu()) * labels.size(0)
        total_primary += float(primary_loss.detach().cpu()) * labels.size(0)
        total_reg += float(outputs.reg_loss.detach().cpu()) * labels.size(0)

        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    y_true = np.vstack(all_labels)
    y_logits = np.vstack(all_logits)
    macro_auc, micro_auc, macro_ap, recall = compute_metrics(y_true, y_logits)

    n = len(loader.dataset)
    return EpochMetrics(
        loss=total_loss / n,
        primary=total_primary / n,
        reg=total_reg / n,
        macro_auc=macro_auc,
        micro_auc=micro_auc,
        macro_ap=macro_ap,
        recall=recall,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MultiModN-style SSM on GeoPlant.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--species-count", type=int, default=342)
    parser.add_argument("--species-vocab", type=Path, default=Path("data/species_vocab.json"))
    parser.add_argument("--state-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pos-weight-cap", type=float, default=50.0)
    parser.add_argument("--lambda-reg", type=float, default=0.05)
    parser.add_argument("--use-sampler", action="store_true")
    parser.add_argument("--loss-type", type=str, choices=["bce", "cb_focal", "logit_adj"], default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--cb-beta", type=float, default=0.999)
    parser.add_argument("--logit-tau", type=float, default=1.0)
    parser.add_argument("--no-env", action="store_true", help="Disable env/climate modality.")
    parser.add_argument("--no-ts", action="store_true", help="Disable time-series modality.")
    parser.add_argument("--no-img", action="store_true", help="Disable image modality.")
    parser.add_argument("--project", type=str, default="geo-plant-ssm")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"], default="offline")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--unfreeze-img", action="store_true", help="Fine-tune image backbone instead of freezing.")
    parser.add_argument("--max-train", type=int, default=None, help="Optional cap on train samples for quick runs.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    seed_everything(args.seed)

    use_env = not args.no_env
    use_ts = not args.no_ts
    use_img = not args.no_img

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata_train = args.data_root / "PA_metadata_train.csv"
    env_path = args.data_root / "EnvironmentalValues/Climate/Average 1981-2010/PA-train-bioclimatic.csv"
    ts_root = args.data_root / "SateliteTimeSeries-Landsat/values"
    rgb_root = args.data_root / "SatelitePatches/PA-train-RGB"

    vocab, freq = build_species_vocab(metadata_train, args.species_count, args.species_vocab)
    labels, label_sids = build_labels(metadata_train, vocab)
    label_density = labels.mean(axis=0)
    print(
        f"Built label matrix: {labels.shape[0]} surveys × {labels.shape[1]} species "
        f"(pos rate min/median/max={label_density.min():.4f}/{np.median(label_density):.4f}/{label_density.max():.4f})"
    )

    env = load_env_features(env_path)
    ts, ts_sids = load_time_series(ts_root, "train")

    common_ids = set(label_sids) & set(env.index.tolist()) & set(ts_sids)
    missing = len(label_sids) - len(common_ids)
    if missing > 0:
        print(f"Skipping {missing} surveys missing at least one modality.")
    if use_img:
        common_ids = filter_ids_with_images(sorted(common_ids), rgb_root)
    else:
        common_ids = sorted(common_ids)

    train_ids, val_ids = split_ids(sorted(common_ids), args.val_ratio, args.seed)
    if args.max_train:
        train_ids = train_ids[: args.max_train]

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
        image_size=args.image_size,
        use_images=use_img,
    )
    val_ds = GeoPlantDataset(
        survey_ids=val_ids,
        labels=val_labels,
        env=env,
        ts=ts,
        ts_sids=ts_sids,
        rgb_root=rgb_root,
        image_size=args.image_size,
        use_images=use_img,
    )

    sampler = make_weighted_sampler(train_labels) if args.use_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    pos_weight = build_pos_weight(train_labels, cap=args.pos_weight_cap).to(device)
    class_priors = build_class_priors(train_labels)
    cb_alpha = build_cb_alpha(train_labels, beta=args.cb_beta)

    def make_loss() -> Callable[[Tensor, Tensor], Tensor]:
        if args.loss_type == "bce":
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            def loss_fn(logits: Tensor, labels: Tensor) -> Tensor:
                return criterion(logits, labels)

            return loss_fn

        if args.loss_type == "cb_focal":
            alpha = cb_alpha
            gamma = args.focal_gamma

            def loss_fn(logits: Tensor, labels: Tensor) -> Tensor:
                return cb_focal_loss(logits, labels, alpha=alpha, gamma=gamma)

            return loss_fn

        if args.loss_type == "logit_adj":
            logit_prior = torch.log(class_priors / (1 - class_priors))
            tau = args.logit_tau

            def loss_fn(logits: Tensor, labels: Tensor) -> Tensor:
                return logit_adjusted_loss(logits, labels, logit_prior=logit_prior, tau=tau)

            return loss_fn

        raise ValueError(f"Unknown loss type: {args.loss_type}")

    loss_fn = make_loss()

    model = StateSpaceModel(
        num_species=args.species_count,
        state_dim=args.state_dim,
        env_dim=train_ds.env.shape[1],
        ts_channels=train_ds.ts.shape[2],
        img_freeze_backbone=not args.unfreeze_img,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

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

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            lambda_reg=args.lambda_reg,
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
            lambda_reg=args.lambda_reg,
            use_env=use_env,
            use_ts=use_ts,
            use_img=use_img,
            optimizer=None,
            scaler=None,
        )

        log_payload = {
            "epoch": epoch,
            "train/loss": train_metrics.loss,
            "train/primary_loss": train_metrics.primary,
            "train/reg": train_metrics.reg,
            "train/macro_auc": train_metrics.macro_auc,
            "train/micro_auc": train_metrics.micro_auc,
            "train/macro_ap": train_metrics.macro_ap,
            "train/recall": train_metrics.recall,
            "val/loss": val_metrics.loss,
            "val/primary_loss": val_metrics.primary,
            "val/reg": val_metrics.reg,
            "val/macro_auc": val_metrics.macro_auc,
            "val/micro_auc": val_metrics.micro_auc,
            "val/macro_ap": val_metrics.macro_ap,
            "val/recall": val_metrics.recall,
        }
        wandb.log(log_payload)

        print(
            f"[{epoch}/{args.epochs}] "
            f"train_loss={train_metrics.loss:.4f} val_loss={val_metrics.loss:.4f} "
            f"val_macro_auc={val_metrics.macro_auc:.3f} val_recall={val_metrics.recall:.3f}"
        )

    wandb.finish()


if __name__ == "__main__":
    main()
