import argparse
import os
import time
from typing import List, Optional, Tuple

import joblib
import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
from tqdm import tqdm

try:
    from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
except ImportError:  # pragma: no cover - optional dependency
    STATUS_OK = None
    Trials = None
    fmin = None
    hp = None
    tpe = None


def compute_time_series_features(ts: np.ndarray) -> np.ndarray:
    """
    Compute simple statistics per Landsat channel to create fixed-length features.
    Expected ts shape: (num_samples, time_steps, num_channels).
    """
    if ts.ndim != 3:
        raise ValueError(f"Expected time series with 3 dims, got shape {ts.shape}")
    num_samples, time_steps, num_channels = ts.shape
    x = np.arange(time_steps, dtype=np.float64)
    x_mean = x.mean()
    denom = np.maximum(((x - x_mean) ** 2).sum(), 1e-6)

    channel_features: List[np.ndarray] = []
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
        per_channel = np.stack(
            [mean, std, min_v, max_v, q25, q50, q75, first, last, delta, slope], axis=1
        )
        channel_features.append(per_channel)
    return np.concatenate(channel_features, axis=1)


def prepare_image_model(
    device: torch.device, backbone: str = "resnet50"
) -> Tuple[torch.nn.Module, transforms.Compose]:
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
) -> np.ndarray:
    """
    Convert RGB patches into ResNet embeddings.
    images shape: (num_samples, C, H, W) or (num_samples, H, W, C).
    Returns array of shape (num_samples, feature_dim).
    """
    feats: List[np.ndarray] = []
    with torch.no_grad():
        iterator = tqdm(
            range(0, len(images), batch_size),
            desc="Image embeddings",
            unit="batch",
        )
        for start in iterator:
            batch = images[start : start + batch_size]
            tensors = []
            for img in batch:
                if img.ndim == 3 and img.shape[0] in (1, 3, 4, 7):  # likely CHW
                    img_hwc = np.transpose(img, (1, 2, 0))
                else:  # assume HWC
                    img_hwc = img
                if img_hwc.shape[-1] == 1:
                    img_hwc = np.repeat(img_hwc, 3, axis=-1)
                elif img_hwc.shape[-1] > 3:
                    img_hwc = img_hwc[..., :3]
                pil_img = Image.fromarray(img_hwc.astype(np.uint8))
                tensors.append(transform(pil_img))
            batch_tensor = torch.stack(tensors).to(device)
            outputs = model(batch_tensor)
            batch_feats = outputs.flatten(1).cpu().numpy()
            feats.append(batch_feats)
    return np.concatenate(feats, axis=0)


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    k = min(k, scores.shape[1])
    idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    hits = labels[np.arange(labels.shape[0])[:, None], idx]
    return float(hits.sum(axis=1).mean() / max(k, 1))


def compute_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float, k: int):
    labels_bin = (labels > 0.5).astype(int)
    preds = (scores >= threshold).astype(int)
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
    except ValueError:
        metrics["auprc_micro"] = float("nan")
        metrics["auprc_macro"] = float("nan")
        metrics["map_macro"] = float("nan")
    metrics["precision_at_k"] = precision_at_k(scores, labels_bin, k=k)
    return metrics


def evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    threshold: float,
    k: int,
) -> dict:
    model.fit(X_train, y_train)
    train_scores = get_scores(model, X_train)
    val_scores = get_scores(model, X_val)
    train_metrics = compute_metrics(train_scores, y_train, threshold=threshold, k=k)
    val_metrics = compute_metrics(val_scores, y_val, threshold=threshold, k=k)
    return {"train": train_metrics, "val": val_metrics}


def run_hyperopt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args,
    auto_pos_weight: Optional[float],
):
    if STATUS_OK is None:
        raise ImportError("hyperopt is required for --hyperopt; install it before running.")

    metric_key = args.hyperopt_metric
    allowed_metrics = {
        "f1_micro",
        "f1_macro",
        "precision_micro",
        "recall_micro",
        "precision_macro",
        "recall_macro",
        "auroc_micro",
        "auroc_macro",
        "auprc_micro",
        "auprc_macro",
        "map_macro",
        "precision_at_k",
    }
    if metric_key not in allowed_metrics:
        raise ValueError(
            f"Unsupported hyperopt metric '{metric_key}'. Must be one of {sorted(allowed_metrics)}"
        )

    def metric_from_metrics(metrics: dict) -> float:
        if metric_key not in metrics:
            raise ValueError(f"Metric '{metric_key}' missing from computed metrics.")
        return float(metrics[metric_key])

    # Define search space per model
    if args.model == "logreg":
        space = {
            "C": hp.loguniform("C", np.log(1e-3), np.log(10.0)),
            "pos_weight": hp.loguniform("pos_weight", np.log(0.5), np.log(20.0)),
            "pca_components": hp.choice("pca_components", [None, 64, 128, 256]),
            "threshold": hp.uniform("threshold", 0.1, 0.7),
            "tol": hp.loguniform("tol", np.log(1e-5), np.log(5e-3)),
        }
    elif args.model == "random_forest":
        space = {
            "trees": hp.quniform("trees", 200, 600, 50),
            "max_depth": hp.choice("max_depth", [None, 8, 12, 16, 20]),
            "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 10, 1),
            "pca_components": hp.choice("pca_components", [None, 64, 128, 256]),
            "threshold": hp.uniform("threshold", 0.1, 0.7),
        }
    else:  # hgbt
        space = {
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
            "max_depth": hp.choice("max_depth", [None, 6, 10, 14]),
            "max_leaf_nodes": hp.choice("max_leaf_nodes", [None, 31, 63, 127]),
            "l2_reg": hp.loguniform("l2_reg", np.log(1e-4), np.log(1.0)),
            "pca_components": hp.choice("pca_components", [None, 64, 128, 256]),
            "threshold": hp.uniform("threshold", 0.1, 0.7),
        }

    trials = Trials()
    rstate = np.random.default_rng(args.hyperopt_seed)

    def objective(space_params):
        pca_components = space_params.get("pca_components")
        if pca_components is not None:
            pca_components = int(pca_components)
        threshold = float(space_params.get("threshold", args.threshold))
        model_kwargs = dict(
            model_name=args.model,
            n_jobs=args.n_jobs,
            seed=args.seed,
            c=float(space_params.get("C", args.c)),
            trees=int(space_params.get("trees", args.trees)),
            depth=None
            if space_params.get("max_depth", args.max_depth) is None
            else int(space_params.get("max_depth", args.max_depth)),
            verbose=args.verbose,
            max_iter=args.max_iter,
            tol=float(space_params.get("tol", args.tol)),
            pos_class_weight=auto_pos_weight if args.model == "logreg" else None,
            pca_components=pca_components,
            pca_random_state=args.seed,
            min_samples_leaf=int(space_params.get("min_samples_leaf", args.min_samples_leaf)),
            hgbt_learning_rate=float(space_params.get("learning_rate", args.hgbt_learning_rate)),
            hgbt_max_depth=None
            if space_params.get("max_depth", args.hgbt_max_depth) is None
            else int(space_params.get("max_depth", args.hgbt_max_depth)),
            hgbt_max_leaf_nodes=None
            if space_params.get("max_leaf_nodes", args.hgbt_max_leaf_nodes) is None
            else int(space_params.get("max_leaf_nodes", args.hgbt_max_leaf_nodes)),
            hgbt_l2_reg=float(space_params.get("l2_reg", args.hgbt_l2_reg)),
        )
        # Allow overriding pos weight in space for logreg
        if args.model == "logreg":
            model_kwargs["pos_class_weight"] = float(
                space_params.get("pos_weight", auto_pos_weight)
            )

        clf = build_model(**model_kwargs)
        clf.fit(X_train, y_train)
        val_scores = get_scores(clf, X_val)
        val_metrics = compute_metrics(val_scores, y_val, threshold=threshold, k=args.precision_at_k)
        metric_val = metric_from_metrics(val_metrics)
        loss = -metric_val
        return {
            "loss": loss,
            "status": STATUS_OK,
            "metric": metric_val,
            "threshold": threshold,
            "params": space_params,
        }

    outer = tqdm(range(args.hyperopt_max_evals), desc="Hyperopt", unit="trial")
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
        outer.set_postfix(best_metric=best_metric)

    if not trials.trials:
        raise RuntimeError("Hyperopt did not run any trials.")
    best_trial = min(trials.trials, key=lambda t: t["result"]["loss"])
    best_params = best_trial["result"]["params"]
    best_threshold = best_trial["result"]["threshold"]

    # Rebuild with best params and evaluate both train/val
    pca_components = best_params.get("pca_components")
    if pca_components is not None:
        pca_components = int(pca_components)
    model_kwargs = dict(
        model_name=args.model,
        n_jobs=args.n_jobs,
        seed=args.seed,
        c=float(best_params.get("C", args.c)),
        trees=int(best_params.get("trees", args.trees)),
        depth=None
        if best_params.get("max_depth", args.max_depth) is None
        else int(best_params.get("max_depth", args.max_depth)),
        verbose=args.verbose,
        max_iter=args.max_iter,
        tol=float(best_params.get("tol", args.tol)),
        pos_class_weight=auto_pos_weight if args.model == "logreg" else None,
        pca_components=pca_components,
        pca_random_state=args.seed,
        min_samples_leaf=int(best_params.get("min_samples_leaf", args.min_samples_leaf)),
        hgbt_learning_rate=float(best_params.get("learning_rate", args.hgbt_learning_rate)),
        hgbt_max_depth=None
        if best_params.get("max_depth", args.hgbt_max_depth) is None
        else int(best_params.get("max_depth", args.hgbt_max_depth)),
        hgbt_max_leaf_nodes=None
        if best_params.get("max_leaf_nodes", args.hgbt_max_leaf_nodes) is None
        else int(best_params.get("max_leaf_nodes", args.hgbt_max_leaf_nodes)),
        hgbt_l2_reg=float(best_params.get("l2_reg", args.hgbt_l2_reg)),
    )
    if args.model == "logreg":
        model_kwargs["pos_class_weight"] = float(best_params.get("pos_weight", auto_pos_weight))

    best_model = build_model(**model_kwargs)
    best_model.fit(X_train, y_train)
    train_scores = get_scores(best_model, X_train)
    val_scores = get_scores(best_model, X_val)
    train_metrics = compute_metrics(
        train_scores, y_train, threshold=best_threshold, k=args.precision_at_k
    )
    val_metrics = compute_metrics(
        val_scores, y_val, threshold=best_threshold, k=args.precision_at_k
    )

    return {
        "best_model": best_model,
        "best_threshold": best_threshold,
        "best_params": best_params,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "trials": trials,
    }


def get_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if isinstance(probs, list):  # Some multi-output estimators return a list
            probs = np.stack([p[:, 1] for p in probs], axis=1)
        return probs
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise ValueError("Model does not provide predict_proba or decision_function.")


def build_model(
    model_name: str,
    n_jobs: int,
    seed: int,
    c: float,
    trees: int,
    depth: int,
    verbose: int,
    max_iter: int,
    tol: float,
    pos_class_weight: Optional[float],
    pca_components: Optional[int],
    pca_random_state: int,
    min_samples_leaf: int,
    hgbt_learning_rate: float,
    hgbt_max_depth: Optional[int],
    hgbt_max_leaf_nodes: Optional[int],
    hgbt_l2_reg: float,
):
    if model_name == "logreg":
        if pos_class_weight is None:
            class_weight = "balanced"
        else:
            class_weight = {0: 1.0, 1: float(pos_class_weight)}
        base = LogisticRegression(
            max_iter=max_iter,
            solver="saga",
            C=c,
            class_weight=class_weight,
            verbose=verbose,
            tol=tol,
        )
        steps = [("scaler", StandardScaler())]
        if pca_components is not None:
            steps.append((
                "pca",
                PCA(
                    n_components=pca_components,
                    whiten=False,
                    random_state=pca_random_state,
                ),
            ))
        steps.append(("clf", OneVsRestClassifier(base, n_jobs=n_jobs)))
        return Pipeline(steps)
    if model_name == "random_forest":
        base = RandomForestClassifier(
            n_estimators=trees,
            max_depth=depth,
            n_jobs=n_jobs,
            class_weight="balanced_subsample",
            random_state=seed,
            verbose=verbose,
            min_samples_leaf=min_samples_leaf,
        )
        steps = []
        if pca_components is not None:
            steps.append(("scaler", StandardScaler()))
            steps.append((
                "pca",
                PCA(
                    n_components=pca_components,
                    whiten=False,
                    random_state=pca_random_state,
                ),
            ))
        steps.append(("clf", OneVsRestClassifier(base, n_jobs=n_jobs)))
        return Pipeline(steps)
    if model_name == "hgbt":
        base = HistGradientBoostingClassifier(
            learning_rate=hgbt_learning_rate,
            max_depth=hgbt_max_depth,
            max_leaf_nodes=hgbt_max_leaf_nodes,
            l2_regularization=hgbt_l2_reg,
            random_state=seed,
            verbose=verbose,
        )
        steps = [("scaler", StandardScaler())]
        if pca_components is not None:
            steps.append((
                "pca",
                PCA(
                    n_components=pca_components,
                    whiten=False,
                    random_state=pca_random_state,
                ),
            ))
        steps.append(("clf", OneVsRestClassifier(base, n_jobs=n_jobs)))
        return Pipeline(steps)
    raise ValueError(f"Unsupported model: {model_name}")


def build_features(
    npz_path: str,
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int,
    use_images: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path) as data:
        missing = [key for key in ("env", "landsat", "labels") if key not in data]
        if missing:
            raise KeyError(f"Missing keys in {npz_path}: {missing}")

        env = np.nan_to_num(data["env"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        ts = np.nan_to_num(data["landsat"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        labels = data["labels"].astype(np.float32)

        print(f"Computing time-series stats for {ts.shape[0]} samples...")
        features = [env, compute_time_series_features(ts).astype(np.float32)]
        if use_images:
            if "images" not in data:
                raise KeyError("Requested image features but 'images' not found in the dataset.")
            imgs = data["images"]
            img_feats = extract_image_features(
                imgs, model=model, transform=transform, device=device, batch_size=batch_size
            ).astype(np.float32)
            features.append(img_feats)
    X = np.concatenate(features, axis=1)
    return X, labels


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Classical ML baselines for multi-label species prediction.",
    )
    parser.add_argument("--train", type=str, required=True, help="Path to training npz file.")
    parser.add_argument("--test", type=str, required=True, help="Path to test/validation npz file.")
    parser.add_argument(
        "--model",
        type=str,
        default="logreg",
        choices=["logreg", "random_forest", "hgbt"],
        help="Which classical model to train.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold.")
    parser.add_argument("--precision-at-k", type=int, default=5, help="k for precision@k.")
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
        "--c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for logistic regression.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Minimum samples per leaf for tree-based models.",
    )
    parser.add_argument(
        "--hgbt-learning-rate", type=float, default=0.1, help="Learning rate for HGBT."
    )
    parser.add_argument(
        "--hgbt-max-depth", type=int, default=None, help="Max depth for HGBT (None=unbounded)."
    )
    parser.add_argument(
        "--hgbt-max-leaf-nodes",
        type=int,
        default=None,
        help="Max leaf nodes for HGBT (None=unbounded).",
    )
    parser.add_argument(
        "--hgbt-l2-reg", type=float, default=0.0, help="L2 regularization for HGBT."
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help="If set, applies PCA to all features before classification with this many components.",
    )
    parser.add_argument("--trees", type=int, default=300, help="Number of trees for random forest.")
    parser.add_argument("--max-depth", type=int, default=None, help="Max depth for random forest.")
    parser.add_argument(
        "--n-jobs", type=int, default=os.cpu_count() or 4, help="Parallel workers for sklearn."
    )
    parser.add_argument(
        "--max-iter", type=int, default=400, help="Max iterations for logistic regression."
    )
    parser.add_argument(
        "--tol", type=float, default=1e-3, help="Tolerance for logistic regression convergence."
    )
    parser.add_argument(
        "--pos-class-weight",
        type=float,
        default=None,
        help="Positive class weight multiplier for logistic regression. Default auto-computes from label prevalence.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--save-model", type=str, default=None, help="Optional path to persist the trained model."
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Verbosity for sklearn estimators (e.g., iteration logs for logreg, tree build logs for RF).",
    )
    parser.add_argument(
        "--hyperopt", action="store_true", help="Run Hyperopt search instead of a single fit."
    )
    parser.add_argument(
        "--hyperopt-max-evals", type=int, default=25, help="Number of Hyperopt trials."
    )
    parser.add_argument(
        "--hyperopt-metric",
        type=str,
        default="auprc_macro",
        help="Metric key from compute_metrics to optimize during Hyperopt.",
    )
    parser.add_argument("--hyperopt-seed", type=int, default=42, help="Random seed for Hyperopt.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_images = not args.no_images
    if use_images:
        image_model, image_transform = prepare_image_model(device, backbone=args.image_backbone)
    else:
        image_model, image_transform = None, None

    print(
        f"Building features (images={'on' if use_images else 'off'}, "
        f"backbone={args.image_backbone}) on device {device}..."
    )
    X_train, y_train = build_features(
        args.train,
        model=image_model,
        transform=image_transform,
        device=device,
        batch_size=args.image_batch_size,
        use_images=use_images,
    )
    X_test, y_test = build_features(
        args.test,
        model=image_model,
        transform=image_transform,
        device=device,
        batch_size=args.image_batch_size,
        use_images=use_images,
    )

    print(
        f"Feature shapes -> train: {X_train.shape}, test: {X_test.shape}, labels: {y_train.shape}"
    )

    pca_components = args.pca_components
    if pca_components is not None:
        pca_components = max(1, min(pca_components, X_train.shape[1]))
        print(f"Using PCA with n_components={pca_components}")

    auto_pos_weight = args.pos_class_weight
    if args.model == "logreg" and args.pos_class_weight is None:
        pos = np.clip(y_train.sum(axis=0), 1e-6, None)
        neg = y_train.shape[0] - pos
        ratios = neg / pos
        auto_pos_weight = float(np.clip(np.nanmedian(ratios), 1.0, 50.0))
        print(f"Auto positive weight (median neg/pos ratio): {auto_pos_weight:.2f}")

    if args.hyperopt:
        print(f"Running Hyperopt for metric '{args.hyperopt_metric}'...")
        hopt_result = run_hyperopt(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            args=args,
            auto_pos_weight=auto_pos_weight,
        )
        best_model = hopt_result["best_model"]
        best_threshold = hopt_result["best_threshold"]
        train_metrics = hopt_result["train_metrics"]
        test_metrics = hopt_result["val_metrics"]
        print(f"Best threshold: {best_threshold:.4f}")
        print("Best params:", hopt_result["best_params"])
        best_metric_val = test_metrics.get(args.hyperopt_metric, float("nan"))
        print(f"Best {args.hyperopt_metric}: {best_metric_val:.4f}")
    else:
        clf = build_model(
            model_name=args.model,
            n_jobs=args.n_jobs,
            seed=args.seed,
            c=args.c,
            trees=args.trees,
            depth=args.max_depth,
            verbose=args.verbose,
            max_iter=args.max_iter,
            tol=args.tol,
            pos_class_weight=auto_pos_weight,
            pca_components=pca_components,
            pca_random_state=args.seed,
            min_samples_leaf=args.min_samples_leaf,
            hgbt_learning_rate=args.hgbt_learning_rate,
            hgbt_max_depth=args.hgbt_max_depth,
            hgbt_max_leaf_nodes=args.hgbt_max_leaf_nodes,
            hgbt_l2_reg=args.hgbt_l2_reg,
        )

        print("Fitting model...")
        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_time = time.time() - t0
        print(f"Fit finished in {fit_time:.1f} seconds.")
        train_scores = get_scores(clf, X_train)
        test_scores = get_scores(clf, X_test)

        train_metrics = compute_metrics(
            train_scores, y_train, threshold=args.threshold, k=args.precision_at_k
        )
        test_metrics = compute_metrics(
            test_scores, y_test, threshold=args.threshold, k=args.precision_at_k
        )

        best_model = clf
        best_threshold = args.threshold

    def format_metrics(split: str, metrics: dict):
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
            "precision_at_k",
        ]
        parts = [f"{split}"]
        for key in ordered:
            if key in metrics:
                parts.append(f"{key}: {metrics[key]:.4f}")
        return " | ".join(parts)

    print(format_metrics("train", train_metrics))
    print(format_metrics("test", test_metrics))

    if args.save_model:
        payload = {"model": best_model, "threshold": best_threshold}
        joblib.dump(payload, args.save_model)
        print(f"Saved model to {args.save_model}")


if __name__ == "__main__":
    main()
