import argparse
import json
import os
from typing import List, Optional, Tuple

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torchvision import transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
from tqdm import tqdm

try:
    from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "hyperopt is required for threshold tuning; install it via `pip install hyperopt`."
    ) from exc


def compute_time_series_features(ts: np.ndarray) -> np.ndarray:
    if ts.ndim != 3:
        raise ValueError(f"Expected time series with 3 dims, got shape {ts.shape}")
    num_samples, time_steps, num_channels = ts.shape
    x = np.arange(time_steps, dtype=np.float64)
    x_mean = x.mean()
    denom = np.maximum(((x - x_mean) ** 2).sum(), 1e-6)

    feats: List[np.ndarray] = []
    for c in range(num_channels):
        data = ts[:, :, c].astype(np.float64)
        mean = np.nanmean(data, axis=1)
        std = np.nanstd(data, axis=1)
        min_v = np.nanmin(data, axis=1)
        max_v = np.nanmax(data, axis=1)
        q25 = np.nanpercentile(data, 25, axis=1)
        q50 = np.nanpercentile(data, 50, axis=1)
        q75 = np.nanpercentile(data, 75, axis=1)
        first = data[:, 0]
        last = data[:, -1]
        delta = last - first
        centered = data - data.mean(axis=1, keepdims=True)
        slope = ((centered) * (x - x_mean)).sum(axis=1) / denom
        feats.append(
            np.stack(
                [mean, std, min_v, max_v, q25, q50, q75, first, last, delta, slope],
                axis=1,
            )
        )
    return np.concatenate(feats, axis=1)


def prepare_image_model(device: torch.device, backbone: str = "resnet50"):
    backbone = backbone.lower()
    if backbone == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    elif backbone == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported image backbone '{backbone}'. Use resnet18 or resnet50.")
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model, weights.transforms()


def extract_image_features(
    images: np.ndarray,
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int,
    desc: str,
) -> np.ndarray:
    feats: List[np.ndarray] = []
    with torch.no_grad():
        iterator = tqdm(
            range(0, len(images), batch_size),
            desc=f"Image embeddings ({desc})",
            unit="batch",
        )
        for start in iterator:
            batch = images[start : start + batch_size]
            tensors = []
            for img in batch:
                if img.ndim == 3 and img.shape[0] in (1, 3, 4, 7):  # likely CHW
                    img_hwc = np.transpose(img, (1, 2, 0))
                else:
                    img_hwc = img
                if img_hwc.shape[-1] == 1:
                    img_hwc = np.repeat(img_hwc, 3, axis=-1)
                elif img_hwc.shape[-1] > 3:
                    img_hwc = img_hwc[..., :3]
                pil_img = Image.fromarray(img_hwc.astype(np.uint8))
                tensors.append(transform(pil_img))
            batch_tensor = torch.stack(tensors).to(device)
            outputs = model(batch_tensor)
            feats.append(outputs.flatten(1).cpu().numpy())
    return np.concatenate(feats, axis=0)


def build_features(
    npz_path: str,
    model: Optional[torch.nn.Module],
    transform: Optional[transforms.Compose],
    device: torch.device,
    batch_size: int,
    use_images: bool,
    desc: str,
) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path) as data:
        missing = [k for k in ("env", "landsat", "labels") if k not in data]
        if missing:
            raise KeyError(f"Missing keys in {npz_path}: {missing}")
        env = np.nan_to_num(data["env"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        ts = np.nan_to_num(data["landsat"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        labels = data["labels"].astype(np.float32)

        print(f"Computing time-series stats for {desc} ({ts.shape[0]} samples)...")
        features = [env, compute_time_series_features(ts).astype(np.float32)]
        if use_images:
            if "images" not in data:
                raise KeyError("Requested image features but 'images' not found in the dataset.")
            imgs = data["images"]
            img_feats = extract_image_features(
                imgs,
                model=model,
                transform=transform,
                device=device,
                batch_size=batch_size,
                desc=desc,
            ).astype(np.float32)
            features.append(img_feats)
    X = np.concatenate(features, axis=1)
    return X, labels


def get_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if isinstance(probs, list):
            probs = np.stack([p[:, 1] for p in probs], axis=1)
        return probs
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise ValueError("Model does not provide predict_proba or decision_function.")


def find_best_thresholds(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    thresholds = np.full(scores.shape[1], 0.5, dtype=np.float32)
    iterator = tqdm(range(scores.shape[1]), desc="Tuning thresholds", unit="class")
    for c in iterator:
        y_true = labels[:, c]
        y_scores = scores[:, c]
        if y_true.sum() == 0:
            thresholds[c] = 0.5
            continue
        precision, recall, thresh = precision_recall_curve(y_true, y_scores)
        if len(thresh) == 0:
            thresholds[c] = 0.5
            continue
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-8)
        best_idx = int(np.argmax(f1))
        thresholds[c] = float(thresh[best_idx])
    return thresholds


def macro_ap_with_thresholds(
    scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray
) -> float:
    labels_bin = (labels > 0.5).astype(int)
    preds = (scores >= thresholds[None, :]).astype(int)
    mask = labels_bin.sum(axis=0) > 0
    if not mask.any():
        return 0.0
    aps = []
    for c in np.where(mask)[0]:
        try:
            aps.append(average_precision_score(labels_bin[:, c], preds[:, c]))
        except ValueError:
            continue
    return float(np.mean(aps)) if aps else 0.0


def optimize_thresholds_hyperopt(
    scores: np.ndarray,
    labels: np.ndarray,
    max_evals: int,
    seed: int,
    metric_name: str,
) -> Tuple[np.ndarray, Trials]:
    num_classes = scores.shape[1]
    space = {f"t{i}": hp.uniform(f"t{i}", 0.0, 1.0) for i in range(num_classes)}
    trials = Trials()
    rstate = np.random.default_rng(seed)

    def metric_for_thresholds(thr: np.ndarray) -> float:
        labels_bin = (labels > 0.5).astype(int)
        preds = (scores >= thr[None, :]).astype(int)
        if metric_name == "auprc_macro_thresholded":
            try:
                return float(average_precision_score(labels_bin, preds, average="macro"))
            except ValueError:
                return 0.0
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            labels_bin, preds, average="macro", zero_division=0
        )
        prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
            labels_bin, preds, average="micro", zero_division=0
        )
        metrics = {
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "precision_micro": prec_micro,
            "recall_micro": rec_micro,
        }
        if metric_name not in metrics:
            raise ValueError(f"Unsupported metric '{metric_name}' for threshold tuning.")
        return float(metrics[metric_name])

    def objective(params):
        thresholds = np.array([params[f"t{i}"] for i in range(num_classes)], dtype=np.float32)
        val = metric_for_thresholds(thresholds)
        loss = -val
        return {
            "loss": loss,
            "status": STATUS_OK,
            "metric": val,
            "thresholds": thresholds.tolist(),
        }

    outer = tqdm(range(max_evals), desc="Hyperopt", unit="trial")
    for _ in outer:
        fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=len(trials.trials) + 1,
            trials=trials,
            rstate=rstate,
            show_progressbar=False,
        )
        best_loss = min(
            (t["result"]["loss"] for t in trials.trials if np.isfinite(t["result"]["loss"])),
            default=float("inf"),
        )
        best_metric = -best_loss if np.isfinite(best_loss) else float("nan")
        outer.set_postfix(best_metric=best_metric, metric=metric_name)

    if not trials.trials:
        raise RuntimeError("Hyperopt did not run any trials.")
    best_trial = min(trials.trials, key=lambda t: t["result"]["loss"])
    best_thresholds = np.array(best_trial["result"]["thresholds"], dtype=np.float32)
    return best_thresholds, trials


def compute_metrics(scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray):
    labels_bin = (labels > 0.5).astype(int)
    preds = (scores >= thresholds[None, :]).astype(int)

    metrics = {}
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        labels_bin, preds, average="micro", zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        labels_bin, preds, average="macro", zero_division=0
    )
    metrics["precision_micro"] = prec_micro
    metrics["recall_micro"] = rec_micro
    metrics["f1_micro"] = f1_micro
    metrics["precision_macro"] = prec_macro
    metrics["recall_macro"] = rec_macro
    metrics["f1_macro"] = f1_macro

    try:
        metrics["auroc_micro"] = roc_auc_score(labels_bin, scores, average="micro")
        metrics["auroc_macro"] = roc_auc_score(labels_bin, scores, average="macro")
    except ValueError:
        metrics["auroc_micro"] = float("nan")
        metrics["auroc_macro"] = float("nan")

    try:
        metrics["auprc_micro"] = average_precision_score(labels_bin, scores, average="micro")
        metrics["auprc_macro"] = average_precision_score(labels_bin, scores, average="macro")
        metrics["map_macro"] = float(
            np.nanmean(average_precision_score(labels_bin, scores, average=None))
        )
        metrics["auprc_macro_thresholded"] = average_precision_score(
            labels_bin, preds, average="macro"
        )
    except ValueError:
        metrics["auprc_micro"] = float("nan")
        metrics["auprc_macro"] = float("nan")
        metrics["map_macro"] = float("nan")
        metrics["auprc_macro_thresholded"] = float("nan")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Tune per-class thresholds on a validation set for a saved classical model.",
    )
    parser.add_argument("--model", type=str, required=True, help="Path to joblib-saved model.")
    parser.add_argument("--data", type=str, required=True, help="Path to validation npz file.")
    parser.add_argument(
        "--image-batch-size", type=int, default=32, help="Batch size for image encoder."
    )
    parser.add_argument(
        "--image-backbone",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet50"],
        help="CNN backbone for image embeddings.",
    )
    parser.add_argument("--no-images", action="store_true", help="Disable image embeddings.")
    parser.add_argument(
        "--save-thresholds",
        type=str,
        default=None,
        help="Optional path to save thresholds (.npy or .json).",
    )
    parser.add_argument(
        "--hyperopt-max-evals", type=int, default=50, help="Number of Hyperopt trials."
    )
    parser.add_argument("--hyperopt-seed", type=int, default=42, help="Random seed for Hyperopt.")
    parser.add_argument(
        "--metric",
        type=str,
        default="auprc_macro_thresholded",
        choices=[
            "auprc_macro_thresholded",
            "f1_macro",
            "f1_micro",
            "precision_macro",
            "recall_macro",
            "precision_micro",
            "recall_micro",
        ],
        help="Metric to optimize when tuning thresholds.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_images = not args.no_images
    if use_images:
        image_model, image_transform = prepare_image_model(device, backbone=args.image_backbone)
    else:
        image_model, image_transform = None, None

    print("Loading model...")
    clf = joblib.load(args.model)

    print(
        f"Building features for tuning set (images={'on' if use_images else 'off'}, "
        f"backbone={args.image_backbone})..."
    )
    X, y = build_features(
        args.data,
        model=image_model,
        transform=image_transform,
        device=device,
        batch_size=args.image_batch_size,
        use_images=use_images,
        desc="tune",
    )
    print(f"Feature shape: {X.shape}, labels: {y.shape}")

    scores = get_scores(clf, X)
    print("Finding best per-class thresholds (max F1 on validation PR curve)...")
    thresholds_f1 = find_best_thresholds(scores, y)

    print(f"Optimizing thresholds for {args.metric} via Hyperopt...")
    thresholds_hopt, trials = optimize_thresholds_hyperopt(
        scores,
        y,
        max_evals=args.hyperopt_max_evals,
        seed=args.hyperopt_seed,
        metric_name=args.metric,
    )

    base_thresholds = np.full(scores.shape[1], 0.5, dtype=np.float32)
    metrics_default = compute_metrics(scores, y, thresholds=base_thresholds)
    metrics_f1 = compute_metrics(scores, y, thresholds=thresholds_f1)
    metrics_hopt = compute_metrics(scores, y, thresholds=thresholds_hopt)

    def fmt(prefix: str, metrics: dict) -> str:
        ordered = [
            "f1_micro",
            "f1_macro",
            "precision_micro",
            "recall_micro",
            "auprc_micro",
            "auprc_macro",
            "auroc_micro",
            "auroc_macro",
            "map_macro",
            "auprc_macro_thresholded",
        ]
        parts = [prefix]
        for key in ordered:
            if key in metrics:
                parts.append(f"{key}: {metrics[key]:.4f}")
        return " | ".join(parts)

    print(fmt("Default thr=0.5", metrics_default))
    print(fmt("F1 per-class PR", metrics_f1))
    print(fmt(f"Hyperopt ({args.metric})", metrics_hopt))
    print(f"Median tuned threshold (F1 PR): {np.median(thresholds_f1):.3f}")
    print(f"Median tuned threshold (Hyperopt): {np.median(thresholds_hopt):.3f}")

    if args.save_thresholds:
        to_save = thresholds_hopt.tolist()
        if args.save_thresholds.endswith(".json"):
            with open(args.save_thresholds, "w", encoding="utf-8") as f:
                json.dump(to_save, f)
        else:
            np.save(args.save_thresholds, thresholds_hopt)
        print(f"Saved Hyperopt thresholds to {args.save_thresholds}")


if __name__ == "__main__":
    main()
