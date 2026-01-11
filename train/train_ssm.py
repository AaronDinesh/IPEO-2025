import argparse
import os

import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision.models import ResNet50_Weights


try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from src.models.ssm.ssm import StateSpaceModel


class IPEODataset(Dataset):
    def __init__(self, filename: str, img_transform=None):
        self.filename = filename
        npz_file = np.load(self.filename)
        self.env_vars = npz_file["env"]
        self.landsat_timeseries = npz_file["landsat"]
        self.images = npz_file["images"]
        self.labels = npz_file["labels"]
        self.dataset_size = self.labels.shape[0]
        self.num_species = self.labels.shape[1]
        self.env_dim = self.env_vars.shape[1]
        self.img_transform = img_transform
        # Expect shape (batch, time, channels); use last dim as feature size.
        self.landsat_channels = self.landsat_timeseries.shape[-1]
        del npz_file

    def __len__(self):
        return self.dataset_size

    def compute_weights_for_loss(self):
        return (self.labels == 0.0).sum(axis=0) / self.labels.sum(axis=0)

    def get_num_species(self):
        return self.num_species

    def get_env_dim(self):
        return self.env_dim

    def get_landsat_channels(self):
        return self.landsat_channels

    def __getitem__(self, idx: int):
        env_var = self.env_vars[idx, :]
        landsat_data = self.landsat_timeseries[idx, :]
        image_data = self.images[idx, :] if self.img_transform is None else self.img_transform(self.images[idx, :])
        label = self.labels[idx, :]

        return ({"env": env_var, "landsat": landsat_data, "images": image_data}, label)


def main(args):
    load_dotenv()
    training_dataset = IPEODataset(args.train, ResNet50_Weights.DEFAULT.transforms)
    testing_dataset = IPEODataset(args.test, ResNet50_Weights.DEFAULT.transforms)

    train_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(testing_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StateSpaceModel(
        num_species=training_dataset.get_num_species(),
        state_dim=args.state_space_dim,
        env_dim=training_dataset.get_env_dim(),
        time_series_channels=training_dataset.get_landsat_channels(),
        img_freeze_backbone=True,
    )
    model.to(device)

    print("=" * 60)
    print("Model Parameters")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print("=" * 60)
    print(f"Using {device} device")

    print("=" * 60)

    pos_weight = torch.tensor(
        training_dataset.compute_weights_for_loss(), device=device, dtype=torch.float32
    )
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    wandb_run = None

    if args.wandb:
        if wandb is None:
            raise ImportError("wandb is not installed; install it or disable --wandb.")
        tags = [t for t in args.wandb_tags.split(",") if t] if args.wandb_tags else None
        wandb_config = {
            "train_path": args.train,
            "test_path": args.test,
            "batch_size": args.batch_size,
            "state_space_dim": args.state_space_dim,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "threshold": args.threshold,
            "precision_at_k": args.precision_at_k,
            "state_change_threshold": args.state_change_threshold,
            "state_change_reg_weight": args.state_change_reg_weight,
            "device": str(device),
            "num_species": training_dataset.get_num_species(),
            "env_dim": training_dataset.get_env_dim(),
            "landsat_channels": training_dataset.get_landsat_channels(),
            "pos_weight_mean": pos_weight.mean().item(),
            "pos_weight_min": pos_weight.min().item(),
            "pos_weight_max": pos_weight.max().item(),
        }
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=args.wandb_group,
            tags=tags,
            config=wandb_config,
        )
        wandb.watch(model, log="all", log_freq=50)

    best_metric = float("-inf")
    best_path = None
    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    def precision_at_k(probs: torch.Tensor, labels: torch.Tensor, k: int) -> float:
        k = min(k, probs.shape[1])
        topk_idx = torch.topk(probs, k, dim=1).indices
        relevant = labels.gather(1, topk_idx)
        return (relevant.sum(dim=1).float() / k).mean().item()

    def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
        probs = torch.sigmoid(logits)
        labels_bin = labels.int()
        preds = (probs >= threshold).int()

        labels_np = labels_bin.cpu().numpy()
        preds_np = preds.cpu().numpy()
        probs_np = probs.cpu().numpy()

        metrics = {}
        prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
            labels_np, preds_np, average="micro", zero_division=0
        )
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            labels_np, preds_np, average="macro", zero_division=0
        )
        metrics["precision_micro"] = prec_micro
        metrics["recall_micro"] = rec_micro
        metrics["f1_micro"] = f1_micro
        metrics["precision_macro"] = prec_macro
        metrics["recall_macro"] = rec_macro
        metrics["f1_macro"] = f1_macro

        try:
            metrics["auroc_micro"] = roc_auc_score(labels_np, probs_np, average="micro")
            metrics["auroc_macro"] = roc_auc_score(labels_np, probs_np, average="macro")
        except ValueError:
            metrics["auroc_micro"] = float("nan")
            metrics["auroc_macro"] = float("nan")

        try:
            ap_per_class = average_precision_score(labels_np, probs_np, average=None)
            metrics["auprc_micro"] = average_precision_score(labels_np, probs_np, average="micro")
            metrics["auprc_macro"] = average_precision_score(labels_np, probs_np, average="macro")
            metrics["map_macro"] = float(np.nanmean(ap_per_class))
        except ValueError:
            metrics["auprc_micro"] = float("nan")
            metrics["auprc_macro"] = float("nan")
            metrics["map_macro"] = float("nan")

        metrics["precision_at_k"] = precision_at_k(probs, labels_bin, k=args.precision_at_k)
        return metrics

    def run_epoch(loader, train: bool, calc_metrics: bool):
        epoch_loss = 0.0
        cls_loss_sum = 0.0
        reg_loss_sum = 0.0
        state_penalty_sum = 0.0
        num_examples = 0
        if train:
            model.train()
        else:
            model.eval()

        collected_logits = []
        collected_labels = []

        iterator = tqdm(loader, leave=False, desc="train" if train else "eval")
        for batch in iterator:
            (features, labels) = batch
            env = torch.as_tensor(features["env"], device=device, dtype=torch.float32)
            ts = torch.as_tensor(features["landsat"], device=device, dtype=torch.float32)
            img = torch.as_tensor(features["images"], device=device, dtype=torch.float32)
            labels = torch.as_tensor(labels, device=device, dtype=torch.float32)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                outputs = model(env=env, ts=ts, img=img)
                logits = outputs.logits[-1]  # use final step prediction
                cls_loss = criterion(logits, labels)
                # Penalize overly large state transitions beyond a threshold.
                initial_state = model.state_init.expand(env.shape[0], -1)
                all_states = [initial_state] + outputs.states
                state_diffs = [
                    all_states[i + 1] - all_states[i] for i in range(len(outputs.states))
                ]
                if state_diffs:
                    stacked_diffs = torch.stack(state_diffs)  # (steps, batch, state_dim)
                    change_norm = torch.norm(stacked_diffs, dim=-1)
                    excess = torch.relu(change_norm - args.state_change_threshold)
                    state_change_penalty = excess.pow(2).mean()
                else:
                    state_change_penalty = torch.tensor(0.0, device=device)

                loss = (
                    cls_loss
                    + outputs.reg_loss
                    + args.state_change_reg_weight * state_change_penalty
                )

            if train:
                loss.backward()
                optimizer.step()

            batch_size = labels.shape[0]
            epoch_loss += loss.item() * batch_size
            cls_loss_sum += cls_loss.item() * batch_size
            reg_loss_sum += outputs.reg_loss.item() * batch_size
            state_penalty_sum += state_change_penalty.item() * batch_size
            num_examples += batch_size
            iterator.set_postfix(loss=loss.item())
            if calc_metrics:
                collected_logits.append(logits.detach().cpu())
                collected_labels.append(labels.detach().cpu())

        avg_loss = epoch_loss / max(num_examples, 1)
        cls_avg = cls_loss_sum / max(num_examples, 1)
        reg_avg = reg_loss_sum / max(num_examples, 1)
        state_penalty_avg = state_penalty_sum / max(num_examples, 1)
        if train and not calc_metrics:
            return {
                "loss": avg_loss,
                "cls_loss": cls_avg,
                "state_reg_loss": reg_avg,
                "state_change_penalty": state_penalty_avg,
            }
        all_logits = torch.cat(collected_logits) if collected_logits else torch.empty(0)
        all_labels = torch.cat(collected_labels) if collected_labels else torch.empty(0)
        metrics = {
            "loss": avg_loss,
            "cls_loss": cls_avg,
            "state_reg_loss": reg_avg,
            "state_change_penalty": state_penalty_avg,
        }
        if calc_metrics and all_logits.numel() > 0:
            metrics.update(compute_metrics(all_logits, all_labels, threshold=args.threshold))
        return metrics

    ## Main Training loop
    for epoch in tqdm(range(args.epochs), desc="Epochs", total=args.epochs, position=0):
        train_stats = run_epoch(train_loader, train=True, calc_metrics=True)
        with torch.no_grad():
            test_stats = run_epoch(test_loader, train=False, calc_metrics=True)
        log_parts = [
            f"Epoch {epoch + 1}/{args.epochs}",
            f"train_loss: {train_stats['loss']:.4f}",
            f"test_loss: {test_stats.get('loss', float('nan')):.4f}",
        ]
        metric_keys = (
            "f1_micro",
            "f1_macro",
            "precision_micro",
            "recall_micro",
            "auroc_micro",
            "auroc_macro",
            "auprc_micro",
            "auprc_macro",
            "map_macro",
            "precision_at_k",
        )
        for key in metric_keys:
            if key in train_stats:
                log_parts.append(f"train_{key}: {train_stats[key]:.4f}")
        for key in metric_keys:
            if key in test_stats:
                log_parts.append(f"test_{key}: {test_stats[key]:.4f}")
        print(" - ".join(log_parts))
        if wandb_run is not None:
            log_payload = {f"train/{k}": v for k, v in train_stats.items()}
            log_payload.update({f"test/{k}": v for k, v in test_stats.items()})
            log_payload["epoch"] = epoch + 1
            wandb.log(log_payload, step=epoch + 1)

        # Checkpointing
        if args.checkpoint_dir is not None:
            metric_key = args.metric_for_best
            current_metric = test_stats.get(metric_key.split("/", 1)[-1], float("-inf"))
            # Also allow full key lookup (train/..., test/...)
            if metric_key in test_stats:
                current_metric = test_stats[metric_key]
            elif metric_key in train_stats:
                current_metric = train_stats[metric_key]
            is_best = current_metric > best_metric
            if is_best:
                best_metric = current_metric
            if is_best or (args.checkpoint_every and (epoch + 1) % args.checkpoint_every == 0):
                ckpt = {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "train_stats": train_stats,
                    "test_stats": test_stats,
                    "best_metric": best_metric,
                    "metric_key": metric_key,
                }
                fname = f"epoch{epoch + 1}_{'best' if is_best else 'ckpt'}.pt"
                ckpt_path = os.path.join(args.checkpoint_dir, fname)
                torch.save(ckpt, ckpt_path)
                if is_best:
                    best_path = ckpt_path
                if wandb_run is not None:
                    wandb.save(ckpt_path)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="SSM Training Script",
        description="Training Script for the State Space Model",
    )
    # fmt: off
    parser.add_argument("--train", type=str, required=True, help="Path to the training npz file")
    parser.add_argument("--test", type=str, required=True, help="Path to the testing npz file")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch Size to use during training")
    parser.add_argument("--state-space-dim", type=int, default=256, help="Size of the state space vector")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate used during training")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs used during training")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for classification metrics")
    parser.add_argument("--precision-at-k", type=int, default=5, help="k used for precision@k")
    parser.add_argument("--state-change-threshold", type=float, default=1.0, help="Threshold for state change magnitude before penalty applies")
    parser.add_argument("--state-change-reg-weight", type=float, default=0.1, help="Weight for the state change penalty term")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="ssm-training", help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default="aarondinesh2002-epfl", help="Weights & Biases entity/user")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--wandb-group", type=str, default=None, help="Weights & Biases group name")
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated tags for Weights & Biases")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save model checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Save checkpoint every N epochs (0 disables periodic checkpoints)")
    parser.add_argument("--metric-for-best", type=str, default="test/auprc_micro", help="Metric key used to track best model")
    # fmt: on
    main(parser.parse_args())
