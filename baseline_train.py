from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
from xgboost import XGBClassifier

from feature_builder import build_feature_sets

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "features"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "baseline_ensemble_oof"

TASKS = {
    "action": {
        "target": "next_actionId",
        "metric": "macro_f1",
    },
    "point": {
        "target": "next_pointId",
        "metric": "macro_f1",
    },
    "server": {
        "target": "target_serverGetPoint",
        "metric": "auc",
    },
}


@dataclass
class TaskEncoding:
    classes: list[int]
    class_to_index: dict[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-friendly OOF ensemble baseline for AI-CUP.")
    parser.add_argument("--raw-train-path", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--raw-test-path", type=Path, default=ROOT / "data" / "test.csv")
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--train-path", type=Path, default=FEATURE_DIR / "train_transition_features.csv")
    parser.add_argument("--test-path", type=Path, default=FEATURE_DIR / "test_rally_features.csv")
    parser.add_argument("--catalog-path", type=Path, default=FEATURE_DIR / "feature_catalog.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--group-col", type=str, default="rally_uid")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--models", nargs="+", default=["catboost", "xgboost", "lightgbm"])
    parser.add_argument("--skip-feature-engineering", action="store_true")
    return parser.parse_args()


def load_feature_columns(catalog_path: Path) -> list[str]:
    catalog = pd.read_csv(catalog_path)
    return catalog.loc[catalog["role"] == "feature", "column"].tolist()


def load_frames(train_path: Path, test_path: Path, feature_cols: list[str], group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    needed_train_cols = [group_col, "rally_uid", "match"] + feature_cols + [cfg["target"] for cfg in TASKS.values()]
    needed_test_cols = ["rally_uid", "match"] + feature_cols
    train = train[sorted(set(needed_train_cols), key=needed_train_cols.index)].copy()
    test = test[sorted(set(needed_test_cols), key=needed_test_cols.index)].copy()
    return train, test


def build_encoded_matrices(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    x_train = train[feature_cols].copy()
    x_test = test[feature_cols].copy()
    categorical_cols = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(x_train[col])]

    for col in feature_cols:
        if x_train[col].dtype == "bool":
            x_train[col] = x_train[col].astype(int)
            x_test[col] = x_test[col].astype(int)

    for col in categorical_cols:
        combined = pd.concat([x_train[col], x_test[col]], axis=0).astype(str)
        cat = pd.Categorical(combined)
        x_train[col] = pd.Categorical(x_train[col].astype(str), categories=cat.categories).codes
        x_test[col] = pd.Categorical(x_test[col].astype(str), categories=cat.categories).codes

    return x_train, x_test, categorical_cols


def encode_target(y: pd.Series) -> tuple[np.ndarray, TaskEncoding]:
    classes = sorted(pd.Series(y).dropna().astype(int).unique().tolist())
    mapping = {cls: idx for idx, cls in enumerate(classes)}
    encoded = y.astype(int).map(mapping).to_numpy()
    return encoded, TaskEncoding(classes=classes, class_to_index=mapping)


def decode_predictions(indices: np.ndarray, encoding: TaskEncoding) -> np.ndarray:
    classes = np.array(encoding.classes)
    return classes[indices]


def score_predictions(y_true: np.ndarray, y_prob: np.ndarray, metric: str, encoding: TaskEncoding | None = None) -> float:
    if metric == "macro_f1":
        assert encoding is not None
        pred_idx = y_prob.argmax(axis=1)
        pred_labels = decode_predictions(pred_idx, encoding)
        return float(f1_score(y_true, pred_labels, average="macro"))
    if metric == "auc":
        return float(roc_auc_score(y_true, y_prob[:, 1]))
    raise ValueError(f"Unsupported metric: {metric}")


def hill_climb_ensemble(
    y_true: np.ndarray,
    metric_name: str,
    encoding: TaskEncoding | None,
    model_oof: dict[str, np.ndarray],
    model_test: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]], float]:
    model_names = list(model_oof.keys())
    scored = []
    for name in model_names:
        score = score_predictions(y_true, model_oof[name], metric_name, encoding)
        scored.append((name, score))
    scored.sort(key=lambda x: x[1], reverse=True)

    best_name, best_score = scored[0]
    current_oof = model_oof[best_name].copy()
    current_test = model_test[best_name].copy()
    weights = {name: 0.0 for name in model_names}
    weights[best_name] = 1.0

    for candidate_name, _ in scored[1:]:
        candidate_oof = model_oof[candidate_name]
        candidate_test = model_test[candidate_name]
        local_best_score = best_score
        local_best_alpha = 0.0

        for alpha in np.linspace(0.05, 0.95, 19):
            blended_oof = (1.0 - alpha) * current_oof + alpha * candidate_oof
            blended_score = score_predictions(y_true, blended_oof, metric_name, encoding)
            if blended_score > local_best_score:
                local_best_score = blended_score
                local_best_alpha = float(alpha)

        if local_best_alpha > 0:
            current_oof = (1.0 - local_best_alpha) * current_oof + local_best_alpha * candidate_oof
            current_test = (1.0 - local_best_alpha) * current_test + local_best_alpha * candidate_test
            for existing_name in weights:
                weights[existing_name] *= 1.0 - local_best_alpha
            weights[candidate_name] += local_best_alpha
            best_score = local_best_score

    ordered_weights = [{"model": name, "weight": float(weight)} for name, weight in weights.items()]
    return current_oof, current_test, ordered_weights, float(best_score)


def get_model(
    model_name: str,
    task_name: str,
    n_classes: int,
    categorical_feature_indices: list[int],
    random_state: int,
    gpu: bool = True,
) -> Any:
    is_binary = task_name == "server"
    if model_name == "catboost":
        params: dict[str, Any] = {
            "random_seed": random_state,
            "verbose": False,
            "allow_writing_files": False,
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 8,
            "loss_function": "Logloss" if is_binary else "MultiClass",
            "eval_metric": "AUC" if is_binary else "TotalF1:average=Macro",
        }
        if gpu:
            params["task_type"] = "GPU"
            params["devices"] = "0"
        else:
            params["task_type"] = "CPU"
        return CatBoostClassifier(**params), {}

    if model_name == "xgboost":
        params = {
            "random_state": random_state,
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 8,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
            "device": "cuda" if gpu else "cpu",
            "objective": "binary:logistic" if is_binary else "multi:softprob",
            "eval_metric": "auc" if is_binary else "mlogloss",
        }
        if not is_binary:
            params["num_class"] = n_classes
        return XGBClassifier(**params), {}

    if model_name == "lightgbm":
        params = {
            "random_state": random_state,
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "binary" if is_binary else "multiclass",
            "device_type": "gpu" if gpu else "cpu",
            "verbose": -1,
        }
        if gpu:
            params["max_bin"] = 255
        if not is_binary:
            params["num_class"] = n_classes
        return LGBMClassifier(**params), {"categorical_feature": categorical_feature_indices}

    raise ValueError(f"Unknown model: {model_name}")


def fit_with_gpu_fallback(
    model_name: str,
    task_name: str,
    n_classes: int,
    categorical_feature_indices: list[int],
    random_state: int,
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[Any, str]:
    model, extra = get_model(
        model_name,
        task_name,
        n_classes,
        categorical_feature_indices,
        random_state,
        gpu=True,
    )
    try:
        model.fit(x_tr, y_tr, sample_weight=sample_weight, **extra)
        return model, "gpu"
    except Exception as exc:
        print(f"[WARN] {model_name} GPU failed for {task_name}: {exc}")
        model, extra = get_model(
            model_name,
            task_name,
            n_classes,
            categorical_feature_indices,
            random_state,
            gpu=False,
        )
        model.fit(x_tr, y_tr, sample_weight=sample_weight, **extra)
        return model, "cpu"


def ensure_prob_shape(proba: np.ndarray, n_classes: int) -> np.ndarray:
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    if proba.shape[1] != n_classes:
        raise ValueError(f"Probability columns mismatch: expected {n_classes}, got {proba.shape[1]}")
    return proba


def align_probabilities(model: Any, proba: np.ndarray, n_classes: int) -> np.ndarray:
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])

    if proba.shape[1] == n_classes:
        return proba

    model_classes = getattr(model, "classes_", None)
    if model_classes is None:
        raise ValueError(f"Probability columns mismatch: expected {n_classes}, got {proba.shape[1]}")

    aligned = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    for src_idx, cls in enumerate(np.asarray(model_classes, dtype=int).tolist()):
        aligned[:, cls] = proba[:, src_idx]
    return aligned


def build_fold_target(y_encoded: np.ndarray, task_name: str, model_name: str) -> tuple[np.ndarray, np.ndarray]:
    if task_name == "server":
        return y_encoded, np.array([0, 1], dtype=int)

    fold_classes = np.unique(y_encoded)
    # XGBoost requires contiguous class ids in each fit call.
    if model_name == "xgboost":
        local_map = {cls: idx for idx, cls in enumerate(fold_classes.tolist())}
        local_y = np.array([local_map[v] for v in y_encoded], dtype=int)
        return local_y, fold_classes.astype(int)

    return y_encoded, fold_classes.astype(int)


def align_fold_probabilities(proba: np.ndarray, n_classes: int, fold_class_indices: np.ndarray) -> np.ndarray:
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])

    if len(fold_class_indices) == n_classes and np.array_equal(fold_class_indices, np.arange(n_classes)):
        return proba

    aligned = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    for src_idx, cls in enumerate(fold_class_indices.tolist()):
        aligned[:, cls] = proba[:, src_idx]
    return aligned


def train_task_oof(
    task_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x_train_num: pd.DataFrame,
    x_test_num: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    group_col: str,
    n_splits: int,
    model_names: list[str],
    random_state: int,
    out_dir: Path,
    progress_bar: tqdm | None = None,
) -> dict[str, Any]:
    target_col = TASKS[task_name]["target"]
    metric_name = TASKS[task_name]["metric"]
    y_true = train_df[target_col].astype(int).to_numpy()
    y_encoded, encoding = encode_target(train_df[target_col])
    groups = train_df[group_col].to_numpy()
    categorical_feature_indices = [feature_cols.index(col) for col in categorical_cols]
    splitter = GroupKFold(n_splits=n_splits)
    n_classes = len(encoding.classes)

    per_model_oof: dict[str, np.ndarray] = {
        name: np.zeros((len(train_df), n_classes), dtype=np.float64) for name in model_names
    }
    per_model_test: dict[str, np.ndarray] = {
        name: np.zeros((len(test_df), n_classes), dtype=np.float64) for name in model_names
    }
    fold_rows: list[dict[str, Any]] = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(x_train_num, y_encoded, groups), start=1):
        y_tr_global = y_encoded[tr_idx]
        y_va = y_true[va_idx]
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_tr_global)

        for model_name in model_names:
            x_tr = x_train_num.iloc[tr_idx].copy()
            x_va = x_train_num.iloc[va_idx].copy()
            x_te = x_test_num.copy()
            y_tr_model, fold_class_indices = build_fold_target(y_tr_global, task_name, model_name)

            model, device_used = fit_with_gpu_fallback(
                model_name=model_name,
                task_name=task_name,
                n_classes=n_classes,
                categorical_feature_indices=categorical_feature_indices,
                random_state=random_state + fold,
                x_tr=x_tr,
                y_tr=y_tr_model,
                sample_weight=sample_weight,
            )

            raw_val_prob = model.predict_proba(x_va)
            raw_test_prob = model.predict_proba(x_te)
            if model_name == "xgboost" and task_name != "server":
                val_prob = align_fold_probabilities(raw_val_prob, n_classes, fold_class_indices)
                test_prob = align_fold_probabilities(raw_test_prob, n_classes, fold_class_indices)
            else:
                val_prob = align_probabilities(model, raw_val_prob, n_classes)
                test_prob = align_probabilities(model, raw_test_prob, n_classes)
            per_model_oof[model_name][va_idx] = val_prob
            per_model_test[model_name] += test_prob / n_splits

            score = score_predictions(y_va, val_prob, metric_name, encoding)
            fold_rows.append(
                {
                    "task": task_name,
                    "fold": fold,
                    "model": model_name,
                    "device": device_used,
                    "metric": metric_name,
                    "score": score,
                    "train_size": int(len(tr_idx)),
                    "valid_size": int(len(va_idx)),
                }
            )
            print(f"[{task_name}] fold={fold} model={model_name} device={device_used} score={score:.6f}")
            if progress_bar is not None:
                progress_bar.set_postfix(
                    task=task_name,
                    fold=f"{fold}/{n_splits}",
                    model=model_name,
                    score=f"{score:.4f}",
                )
                progress_bar.update(1)

    oof_rows = []
    ensemble_oof = np.zeros((len(train_df), n_classes), dtype=np.float64)
    ensemble_test = np.zeros((len(test_df), n_classes), dtype=np.float64)

    for model_name in model_names:
        model_score = score_predictions(y_true, per_model_oof[model_name], metric_name, encoding)
        oof_rows.append({"task": task_name, "model": model_name, "metric": metric_name, "oof_score": model_score})
        ensemble_oof += per_model_oof[model_name]
        ensemble_test += per_model_test[model_name]

    ensemble_oof /= len(model_names)
    ensemble_test /= len(model_names)
    ensemble_score = score_predictions(y_true, ensemble_oof, metric_name, encoding)
    oof_rows.append({"task": task_name, "model": "ensemble_mean", "metric": metric_name, "oof_score": ensemble_score})

    hill_oof, hill_test, hill_weights, hill_score = hill_climb_ensemble(
        y_true=y_true,
        metric_name=metric_name,
        encoding=encoding,
        model_oof=per_model_oof,
        model_test=per_model_test,
    )
    oof_rows.append({"task": task_name, "model": "ensemble_hillclimb", "metric": metric_name, "oof_score": hill_score})

    final_oof = hill_oof if hill_score >= ensemble_score else ensemble_oof
    final_test = hill_test if hill_score >= ensemble_score else ensemble_test
    final_name = "ensemble_hillclimb" if hill_score >= ensemble_score else "ensemble_mean"
    final_score = hill_score if hill_score >= ensemble_score else ensemble_score

    class_cols = [f"class_{cls}" for cls in encoding.classes]
    oof_df = pd.DataFrame(final_oof, columns=class_cols)
    oof_df.insert(0, "rally_uid", train_df["rally_uid"].to_numpy())
    oof_df.insert(1, "target", y_true)
    oof_df.insert(2, "task", task_name)
    oof_df.to_parquet(out_dir / f"oof_{task_name}.parquet", index=False)

    test_pred_df = pd.DataFrame(final_test, columns=class_cols)
    test_pred_df.insert(0, "rally_uid", test_df["rally_uid"].to_numpy())
    test_pred_df.insert(1, "task", task_name)
    test_pred_df.to_parquet(out_dir / f"test_pred_{task_name}.parquet", index=False)

    pred_idx = final_test.argmax(axis=1)
    pred_labels = decode_predictions(pred_idx, encoding)

    return {
        "task_name": task_name,
        "target_col": target_col,
        "metric": metric_name,
        "encoding": encoding,
        "fold_scores": fold_rows,
        "oof_scores": oof_rows,
        "ensemble_score": final_score,
        "ensemble_name": final_name,
        "ensemble_weights": hill_weights,
        "mean_ensemble_score": ensemble_score,
        "hillclimb_score": hill_score,
        "test_prob": final_test,
        "test_pred_labels": pred_labels,
    }


def build_submission(test_df: pd.DataFrame, results: dict[str, dict[str, Any]], out_dir: Path) -> pd.DataFrame:
    submission = pd.DataFrame({"rally_uid": test_df["rally_uid"].to_numpy()})
    submission["actionId"] = results["action"]["test_pred_labels"].astype(int)
    submission["pointId"] = results["point"]["test_pred_labels"].astype(int)
    submission["serverGetPoint"] = results["server"]["test_prob"][:, 1]
    submission.to_csv(out_dir / "submission.csv", index=False, encoding="utf-8-sig")
    return submission


def _plot_bar(df: pd.DataFrame, x: str, y: str, title: str, path: Path, color: str = "#4C78A8", rotate: int = 0) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(df[x].astype(str), df[y], color=color)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=rotate, ha="right" if rotate else "center")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_grouped_scores(oof_scores: pd.DataFrame, path: Path) -> None:
    tasks = oof_scores["task"].unique().tolist()
    preferred_tail = [m for m in ["ensemble_mean", "ensemble_hillclimb"] if m in oof_scores["model"].unique().tolist()]
    models = [m for m in oof_scores["model"].unique().tolist() if m not in preferred_tail] + preferred_tail
    x = np.arange(len(tasks))
    width = 0.18
    plt.figure(figsize=(11, 5))
    for idx, model in enumerate(models):
        sub = oof_scores[oof_scores["model"] == model].set_index("task").reindex(tasks)
        plt.bar(x + (idx - (len(models) - 1) / 2) * width, sub["oof_score"], width=width, label=model)
    plt.xticks(x, tasks)
    plt.ylabel("score")
    plt.title("OOF scores by task and model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_fold_scores(fold_scores: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    tasks = ["action", "point", "server"]
    colors = {"catboost": "#4C78A8", "xgboost": "#F28E2B", "lightgbm": "#59A14F"}
    for ax, task in zip(axes, tasks):
        sub = fold_scores[fold_scores["task"] == task]
        for model in ["catboost", "xgboost", "lightgbm"]:
            model_sub = sub[sub["model"] == model]
            ax.plot(model_sub["fold"], model_sub["score"], marker="o", label=model, color=colors[model])
        ax.set_title(task)
        ax.set_xlabel("fold")
        ax.set_ylabel("score")
    axes[0].legend()
    fig.suptitle("Fold-level validation stability")
    fig.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _read_oof_classes(oof_path: Path) -> tuple[pd.DataFrame, list[int]]:
    df = pd.read_parquet(oof_path)
    class_cols = [col for col in df.columns if col.startswith("class_")]
    classes = [int(col.replace("class_", "")) for col in class_cols]
    return df, classes


def _plot_classwise_f1(oof_path: Path, title: str, path: Path) -> pd.DataFrame:
    df, classes = _read_oof_classes(oof_path)
    class_cols = [f"class_{cls}" for cls in classes]
    probs = df[class_cols].to_numpy()
    preds = np.array(classes)[probs.argmax(axis=1)]
    y_true = df["target"].to_numpy()
    f1s = f1_score(y_true, preds, labels=classes, average=None, zero_division=0)
    support = pd.Series(y_true).value_counts().reindex(classes, fill_value=0).to_numpy()
    out = pd.DataFrame({"class": classes, "f1": f1s, "support": support}).sort_values("f1")

    plt.figure(figsize=(11, 5))
    plt.bar(out["class"].astype(str), out["f1"], color="#E15759")
    plt.title(title)
    plt.xlabel("class")
    plt.ylabel("F1")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return out


def _plot_server_roc(oof_path: Path, path: Path) -> float:
    df, classes = _read_oof_classes(oof_path)
    pos_col = "class_1" if "class_1" in df.columns else df.columns[-1]
    y_true = df["target"].to_numpy()
    y_prob = df[pos_col].to_numpy()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}", color="#59A14F")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Server task ROC (OOF)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return float(auc)


def generate_training_report(out_dir: Path) -> None:
    report_dir = out_dir / "report"
    plots_dir = report_dir / "plots"
    report_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    oof_scores = pd.read_csv(out_dir / "oof_scores.csv")
    fold_scores = pd.read_csv(out_dir / "fold_scores.csv")

    task_summary = pd.DataFrame(
        [
            {"task": "action", "score": summary["task_scores"]["action_macro_f1"]},
            {"task": "point", "score": summary["task_scores"]["point_macro_f1"]},
            {"task": "server", "score": summary["task_scores"]["server_auc"]},
        ]
    )
    _plot_bar(task_summary, "task", "score", "Ensemble task scores", plots_dir / "01_task_scores.png", color="#4C78A8")
    _plot_grouped_scores(oof_scores, plots_dir / "02_model_vs_ensemble_scores.png")
    _plot_fold_scores(fold_scores, plots_dir / "03_fold_stability.png")
    action_class_f1 = _plot_classwise_f1(out_dir / "oof_action.parquet", "Action class-wise F1 (OOF)", plots_dir / "04_action_class_f1.png")
    point_class_f1 = _plot_classwise_f1(out_dir / "oof_point.parquet", "Point class-wise F1 (OOF)", plots_dir / "05_point_class_f1.png")
    server_auc = _plot_server_roc(out_dir / "oof_server.parquet", plots_dir / "06_server_roc.png")

    weak_action = action_class_f1.head(5)
    weak_point = point_class_f1.head(5)
    def ensemble_row(task: str) -> pd.Series:
        sub = oof_scores[oof_scores["task"] == task]
        ensemble_sub = sub[sub["model"].str.startswith("ensemble_")].sort_values("oof_score", ascending=False)
        return ensemble_sub.iloc[0]

    action_ensemble = ensemble_row("action")
    point_ensemble = ensemble_row("point")
    server_ensemble = ensemble_row("server")
    action_gain = float(
        action_ensemble["oof_score"]
        - oof_scores[(oof_scores["task"] == "action") & (~oof_scores["model"].str.startswith("ensemble_"))]["oof_score"].max()
    )
    point_gain = float(
        point_ensemble["oof_score"]
        - oof_scores[(oof_scores["task"] == "point") & (~oof_scores["model"].str.startswith("ensemble_"))]["oof_score"].max()
    )
    server_gain = float(
        server_ensemble["oof_score"]
        - oof_scores[(oof_scores["task"] == "server") & (~oof_scores["model"].str.startswith("ensemble_"))]["oof_score"].max()
    )

    def format_small_table(df: pd.DataFrame) -> str:
        header = "| class | f1 | support |\n|---:|---:|---:|\n"
        rows = "".join(
            f"| {int(row['class'])} | {row['f1']:.4f} | {int(row['support'])} |\n"
            for _, row in df.iterrows()
        )
        return header + rows

    report = f"""# Baseline 訓練結果報告

## 整體結果

- Overall Score: `{summary["overall_score"]:.6f}`
- Action Macro F1: `{summary["task_scores"]["action_macro_f1"]:.6f}`
- Point Macro F1: `{summary["task_scores"]["point_macro_f1"]:.6f}`
- Server AUC: `{server_auc:.6f}`
- Final ensemble variants: `action={action_ensemble["model"]}`, `point={point_ensemble["model"]}`, `server={server_ensemble["model"]}`

## 圖表

- `report/plots/01_task_scores.png`
- `report/plots/02_model_vs_ensemble_scores.png`
- `report/plots/03_fold_stability.png`
- `report/plots/04_action_class_f1.png`
- `report/plots/05_point_class_f1.png`
- `report/plots/06_server_roc.png`

## 解讀

- `action` 任務是目前最有競爭力的多分類任務，ensemble 相對最佳單模額外提升 `{action_gain:.6f}`。
- `point` 任務是最大短板，ensemble 雖然仍有提升 `{point_gain:.6f}`，但整體分數仍偏低，代表需要更針對性優化。
- `server` 任務目前主要由 `catboost` 撐住，ensemble 只帶來很小增益 `{server_gain:.6f}`，說明這個任務比較接近上下文訊號主導。

## 最需要優先改善

1. `pointId` 任務。
原因：目前分數最低，且 class-wise F1 會顯示多個類別幾乎沒有被學好。

2. `actionId` 尾端類別。
原因：整體已有一定表現，但仍有弱類別拖累 Macro F1。

3. 驗證穩定性。
原因：`action` 的第 3 fold 明顯低於前兩折，表示資料分布或切分敏感度存在。

## 建議下一步

1. 先做 `match` 分組驗證，檢查目前 `rally_uid` OOF 是否偏樂觀。
2. 為 `pointId` 加上更積極的 class imbalance 策略，例如 class weight、tail upweight、或 two-stage 預測。
3. 針對 `action` 與 `point` 導入更強的交互特徵，例如 `action_point_combo` 的頻次編碼、prefix 統計、或任務間 stacking。
4. 若 baseline 穩定，再加第二層 ensemble，例如對三模型 OOF 做 stacking，而不是只做平均。

## 弱類別觀察

### Action 最弱 5 類

{format_small_table(weak_action)}

### Point 最弱 5 類

{format_small_table(weak_point)}
"""
    (report_dir / "training_report.md").write_text(report, encoding="utf-8")

    html = """<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <title>Baseline Training Dashboard</title>
  <style>
    body { font-family: "Segoe UI", sans-serif; margin: 24px; background: #f7f7f9; color: #222; }
    h1, h2 { margin-bottom: 8px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }
    .card { background: white; border-radius: 10px; padding: 14px; box-shadow: 0 2px 10px rgba(0,0,0,0.07); }
    img { width: 100%; border-radius: 8px; }
  </style>
</head>
<body>
  <h1>Baseline Training Dashboard</h1>
  <div class="grid">
    <div class="card"><h2>Task Scores</h2><img src="plots/01_task_scores.png" alt="task scores"></div>
    <div class="card"><h2>Model vs Ensemble</h2><img src="plots/02_model_vs_ensemble_scores.png" alt="model scores"></div>
    <div class="card"><h2>Fold Stability</h2><img src="plots/03_fold_stability.png" alt="fold stability"></div>
    <div class="card"><h2>Action Class F1</h2><img src="plots/04_action_class_f1.png" alt="action class f1"></div>
    <div class="card"><h2>Point Class F1</h2><img src="plots/05_point_class_f1.png" alt="point class f1"></div>
    <div class="card"><h2>Server ROC</h2><img src="plots/06_server_roc.png" alt="server roc"></div>
  </div>
</body>
</html>
"""
    (report_dir / "dashboard.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_feature_engineering:
        print("[STAGE 1/2] feature engineering from raw data")
        build_feature_sets(args.raw_train_path, args.raw_test_path, args.feature_dir)
        args.train_path = args.feature_dir / "train_transition_features.csv"
        args.test_path = args.feature_dir / "test_rally_features.csv"
        args.catalog_path = args.feature_dir / "feature_catalog.csv"
    else:
        print("[STAGE 1/2] skip feature engineering, use existing feature tables")

    print("[STAGE 2/2] OOF ensemble training")
    feature_cols = load_feature_columns(args.catalog_path)
    train_df, test_df = load_frames(args.train_path, args.test_path, feature_cols, args.group_col)
    x_train_num, x_test_num, categorical_cols = build_encoded_matrices(train_df, test_df, feature_cols)

    all_fold_scores: list[dict[str, Any]] = []
    all_oof_scores: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    total_steps = len(TASKS) * args.n_splits * len(args.models)

    with tqdm(total=total_steps, desc="OOF Ensemble Progress", dynamic_ncols=True, ascii=True) as progress_bar:
        for task_name in TASKS:
            print(f"[INFO] start task={task_name}")
            task_result = train_task_oof(
                task_name=task_name,
                train_df=train_df,
                test_df=test_df,
                x_train_num=x_train_num,
                x_test_num=x_test_num,
                feature_cols=feature_cols,
                categorical_cols=categorical_cols,
                group_col=args.group_col,
                n_splits=args.n_splits,
                model_names=args.models,
                random_state=args.random_state,
                out_dir=args.out_dir,
                progress_bar=progress_bar,
            )
            results[task_name] = task_result
            all_fold_scores.extend(task_result["fold_scores"])
            all_oof_scores.extend(task_result["oof_scores"])

    fold_df = pd.DataFrame(all_fold_scores)
    oof_df = pd.DataFrame(all_oof_scores)
    fold_df.to_csv(args.out_dir / "fold_scores.csv", index=False, encoding="utf-8-sig")
    oof_df.to_csv(args.out_dir / "oof_scores.csv", index=False, encoding="utf-8-sig")

    overall_score = (
        results["action"]["ensemble_score"] * 0.4
        + results["point"]["ensemble_score"] * 0.4
        + results["server"]["ensemble_score"] * 0.2
    )

    summary = {
        "group_col": args.group_col,
        "n_splits": args.n_splits,
        "models": args.models,
        "feature_count": len(feature_cols),
        "categorical_feature_count": len(categorical_cols),
        "task_scores": {
            "action_macro_f1": results["action"]["ensemble_score"],
            "point_macro_f1": results["point"]["ensemble_score"],
            "server_auc": results["server"]["ensemble_score"],
        },
        "ensemble_details": {
            "action": {
                "final_name": results["action"]["ensemble_name"],
                "mean_score": results["action"]["mean_ensemble_score"],
                "hillclimb_score": results["action"]["hillclimb_score"],
                "weights": results["action"]["ensemble_weights"],
            },
            "point": {
                "final_name": results["point"]["ensemble_name"],
                "mean_score": results["point"]["mean_ensemble_score"],
                "hillclimb_score": results["point"]["hillclimb_score"],
                "weights": results["point"]["ensemble_weights"],
            },
            "server": {
                "final_name": results["server"]["ensemble_name"],
                "mean_score": results["server"]["mean_ensemble_score"],
                "hillclimb_score": results["server"]["hillclimb_score"],
                "weights": results["server"]["ensemble_weights"],
            },
        },
        "overall_score": overall_score,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    build_submission(test_df, results, args.out_dir)
    generate_training_report(args.out_dir)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
