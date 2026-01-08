from __future__ import annotations

import argparse
import math
import random

# Ensure local src/ is importable when running as a script
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import wandb
from dotenv import load_dotenv
from PIL import Image
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from src.models.ssm.ssm import StateSpaceModel  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_species_vocab(metadata_path: Path) -> Tuple[Dict[int, int], pd.Series]:
    """Build mapping from speciesId to class index using first appearance order."""

    df = pd.read_csv(metadata_path)
    df["speciesId"] = df["speciesId"].astype(int)
    freq = df["speciesId"].value_counts()

    # Preserve the order of first appearance in the dataset to assign indices.
    unique_ids = pd.unique(df["speciesId"])
    vocab = {int(sid): idx for idx, sid in enumerate(unique_ids)}

    return vocab, freq


def build_labels(metadata_path: Path, vocab: Dict[int, int]) -> Tuple[np.ndarray, List[int]]:
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
        csv_path = (
            ts_root
            / f"PA-{split}-landsat_time_series"
            / f"PA-{split}-landsat_time_series-{band}.csv"
        )
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
        allow_missing_images: bool = True,
    ):
        self.survey_ids = list(survey_ids)
        self.labels = labels
        self.env = env
        self.env_cols = list(env.columns)
        self.ts = ts
        self.ts_index = {sid: idx for idx, sid in enumerate(ts_sids)}
        self.rgb_root = rgb_root
        self.use_images = use_images
        self.allow_missing_images = allow_missing_images
        self._missing_img_warned = False
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.survey_ids)

    def __getitem__(self, idx: int):
        sid = self.survey_ids[idx]
        label = torch.from_numpy(self.labels[idx])

        env_mask = sid in self.env.index
        ts_mask = sid in self.ts_index
        img_mask = False

        if env_mask:
            env_vec = torch.tensor(self.env.loc[sid].to_numpy(np.float32), dtype=torch.float32)
        else:
            env_vec = torch.zeros(len(self.env_cols), dtype=torch.float32)

        if ts_mask:
            ts_idx = self.ts_index[sid]
            ts_tensor = torch.tensor(self.ts[ts_idx], dtype=torch.float32)
        else:
            ts_tensor = torch.zeros(self.ts.shape[1:], dtype=torch.float32)

        if self.use_images:
            img_path = patch_path(self.rgb_root, sid)
            if not img_path.exists():
                if not self.allow_missing_images:
                    raise FileNotFoundError(f"Image not found for surveyId={sid}: {img_path}")
                if not self._missing_img_warned:
                    print(
                        "Warning: missing RGB patches encountered; filling zeros for missing images."
                    )
                    self._missing_img_warned = True
                img_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)
            else:
                img_mask = True
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    img_tensor = self.transform(im)
        else:
            img_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)
            img_mask = False

        return (
            env_vec,
            ts_tensor,
            img_tensor,
            torch.tensor(env_mask, dtype=torch.bool),
            torch.tensor(ts_mask, dtype=torch.bool),
            torch.tensor(img_mask, dtype=torch.bool),
            label,
            sid,
        )


def split_ids(ids: Sequence[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    split = int(len(ids) * (1 - val_ratio))
    return ids[:split], ids[split:]


def build_pos_weight(labels: np.ndarray, eps: float = 1e-6) -> torch.Tensor:
    """Compute pos_weight as mean(count) / count to upweight rarer classes."""

    counts = labels.sum(axis=0)
    mean_count = counts.mean()
    weight = mean_count / (counts + eps)
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


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    args: argparse.Namespace,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
    }
    torch.save(payload, path)


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

    for batch in tqdm(
        loader,
        desc="train" if is_train else "eval",
        total=len(loader),
        leave=True,
        dynamic_ncols=True,
        mininterval=0.1,
    ):
        env, ts, img, env_mask, ts_mask, img_mask, labels, _ = batch
        env = env.to(device)
        ts = ts.to(device)
        img = img.to(device)
        env_mask = env_mask.to(device)
        ts_mask = ts_mask.to(device)
        img_mask = img_mask.to(device)
        labels = labels.to(device)

        with torch.amp.autocast(
            device_type="cuda", enabled=device.type == "cuda" and scaler is not None
        ):
            outputs = model(
                env=env if use_env else None,
                ts=ts if use_ts else None,
                img=img if use_img else None,
                env_mask=env_mask if use_env else None,
                ts_mask=ts_mask if use_ts else None,
                img_mask=img_mask if use_img else None,
            )
            logits = outputs.logits[-1]
            step_losses = []
            for logit, mask in zip(outputs.logits, outputs.step_masks):
                if mask is None:
                    step_losses.append(loss_fn(logit, labels))
                else:
                    valid = mask.bool()
                    if valid.any():
                        step_losses.append(loss_fn(logit[valid], labels[valid]))
                    else:
                        step_losses.append(torch.tensor(0.0, device=logit.device))
            primary_loss = (
                sum(step_losses) / len(step_losses)
                if step_losses
                else torch.tensor(0.0, device=labels.device)
            )
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
    parser.add_argument("--state-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--lambda-reg", type=float, default=0.05)
    parser.add_argument("--use-sampler", action="store_true")
    parser.add_argument("--no-env", action="store_true", help="Disable env/climate modality.")
    parser.add_argument("--no-ts", action="store_true", help="Disable time-series modality.")
    parser.add_argument("--no-img", action="store_true", help="Disable image modality.")
    parser.add_argument(
        "--require-img", action="store_true", help="Error if an image patch is missing."
    )
    parser.add_argument("--project", type=str, default="geo-plant-ssm")
    parser.add_argument("--entity", type=str, default="aarondinesh2002-epfl")
    parser.add_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--run-name", type=str, default=None, help="Optional human-friendly run name for wandb.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--unfreeze-img", action="store_true", help="Fine-tune image backbone instead of freezing."
    )
    parser.add_argument(
        "--max-train", type=int, default=None, help="Optional cap on train samples for quick runs."
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--ckpt-every", type=int, default=10, help="Save checkpoint every N epochs."
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    seed_everything(args.seed)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_macro_auc = float("-inf")
    best_ckpt_path: Optional[Path] = None

    use_env = not args.no_env
    use_ts = not args.no_ts
    use_img = not args.no_img

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata_train = args.data_root / "PA_metadata_train.csv"
    env_path = (
        args.data_root / "EnvironmentalValues/Climate/Average 1981-2010/PA-train-bioclimatic.csv"
    )
    ts_root = args.data_root / "SateliteTimeSeries-Landsat/values"
    rgb_root = args.data_root / "SatelitePatches/PA-train-RGB"

    vocab, _ = build_species_vocab(metadata_train)
    species_count = len(vocab)
    labels, label_sids = build_labels(metadata_train, vocab)
    label_density = labels.mean(axis=0)
    print(
        f"Built label matrix: {labels.shape[0]} surveys × {labels.shape[1]} species "
        f"(pos rate min/median/max={label_density.min():.4f}/{np.median(label_density):.4f}/{label_density.max():.4f})"
    )

    env = load_env_features(env_path)
    ts, ts_sids = load_time_series(ts_root, "train")

    all_ids = set(label_sids)
    missing_env = len(all_ids - set(env.index.tolist()))
    missing_ts = len(all_ids - set(ts_sids))
    if missing_env > 0:
        print(f"{missing_env} surveys missing env features will use zero fill.")
    if missing_ts > 0:
        print(f"{missing_ts} surveys missing time-series will use zero fill.")

    common_ids = sorted(all_ids)
    if use_img and args.require_img:
        common_ids = filter_ids_with_images(common_ids, rgb_root)

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
        allow_missing_images=not args.require_img,
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
        allow_missing_images=not args.require_img,
    )

    sampler = make_weighted_sampler(train_labels) if args.use_sampler else None
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    pos_weight = build_pos_weight(train_labels).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def loss_fn(logits: Tensor, labels: Tensor) -> Tensor:
        return criterion(logits, labels)

    model = StateSpaceModel(
        num_species=species_count,
        state_dim=args.state_dim,
        env_dim=train_ds.env.shape[1],
        ts_channels=train_ds.ts.shape[2],
        img_freeze_backbone=not args.unfreeze_img,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    wandb_config = {**vars(args), "species_count": species_count}

    wandb_mode = args.wandb_mode
    if wandb_mode == "disabled":
        wandb.init(mode="disabled", config=wandb_config, name=args.run_name)
    else:
        wandb.init(
            project=args.project,
            entity=args.entity,
            mode=wandb_mode,
            config=wandb_config,
            name=args.run_name,
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

        if args.ckpt_every > 0 and epoch % args.ckpt_every == 0:
            save_checkpoint(
                args.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt",
                model,
                optimizer,
                scaler,
                epoch,
                args,
            )

        if math.isfinite(val_metrics.macro_auc) and val_metrics.macro_auc > best_macro_auc:
            if best_ckpt_path and best_ckpt_path.exists():
                best_ckpt_path.unlink()
            best_macro_auc = val_metrics.macro_auc
            best_ckpt_path = args.checkpoint_dir / f"HIGHEST_ACCURACY_{epoch}.pt"
            save_checkpoint(best_ckpt_path, model, optimizer, scaler, epoch, args)

    wandb.finish()
    save_checkpoint(
        args.checkpoint_dir / "checkpoint_final.pt", model, optimizer, scaler, args.epochs, args
    )


if __name__ == "__main__":
    main()
