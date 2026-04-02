from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from feature_builder import build_feature_sets
from baseline_train import (
    TASKS,
    align_fold_probabilities,
    align_probabilities,
    build_encoded_matrices,
    build_fold_target,
    cleanup_gpu_memory,
    decode_predictions,
    encode_target,
    load_feature_columns,
    load_frames,
    to_host_numpy,
    to_xgb_device_matrix,
    to_xgb_device_vector,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "features"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "optuna_tuning"

TASK_MODEL = {
    "action": "xgboost",
    "point": "xgboost",
    "server": "catboost",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuning for AI-CUP strongest baseline models.")
    parser.add_argument("--raw-train-path", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--raw-test-path", type=Path, default=ROOT / "data" / "test.csv")
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--train-path", type=Path, default=FEATURE_DIR / "train_transition_features.csv")
    parser.add_argument("--test-path", type=Path, default=FEATURE_DIR / "test_rally_features.csv")
    parser.add_argument("--catalog-path", type=Path, default=FEATURE_DIR / "feature_catalog.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--group-col", type=str, default="rally_uid")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tasks", nargs="+", choices=["action", "point", "server"], default=["action", "point", "server"])
    parser.add_argument("--skip-feature-engineering", action="store_true")
    return parser.parse_args()


def score_predictions(y_true: np.ndarray, y_prob: np.ndarray, metric: str, classes: list[int] | None = None) -> float:
    if metric == "macro_f1":
        assert classes is not None
        pred_idx = y_prob.argmax(axis=1)
        pred_labels = decode_predictions(pred_idx, type("Encoding", (), {"classes": classes})())
        return float(f1_score(y_true, pred_labels, average="macro"))
    if metric == "auc":
        return float(roc_auc_score(y_true, y_prob[:, 1]))
    raise ValueError(metric)


def suggest_params(trial: optuna.Trial, task_name: str) -> dict[str, Any]:
    model_name = TASK_MODEL[task_name]
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 250, 900, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
        }
    return {
        "iterations": trial.suggest_int("iterations", 250, 900, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
    }


def build_model(task_name: str, params: dict[str, Any], random_state: int, n_classes: int) -> Any:
    model_name = TASK_MODEL[task_name]
    if model_name == "xgboost":
        base = {
            "random_state": random_state,
            "tree_method": "hist",
            "device": "cuda",
            "objective": "binary:logistic" if task_name == "server" else "multi:softprob",
            "eval_metric": "auc" if task_name == "server" else "mlogloss",
        }
        if task_name != "server":
            base["num_class"] = n_classes
        base.update(params)
        return XGBClassifier(**base)

    base = {
        "random_seed": random_state,
        "verbose": False,
        "allow_writing_files": False,
        "task_type": "GPU",
        "devices": "0",
        "loss_function": "Logloss" if task_name == "server" else "MultiClass",
        "eval_metric": "AUC" if task_name == "server" else "TotalF1:average=Macro",
    }
    base.update(params)
    return CatBoostClassifier(**base)


def objective_factory(
    task_name: str,
    train_df: pd.DataFrame,
    x_train: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    group_col: str,
    n_splits: int,
    random_state: int,
):
    target_col = TASKS[task_name]["target"]
    metric_name = TASKS[task_name]["metric"]
    y_true = train_df[target_col].astype(int).to_numpy()
    y_encoded, encoding = encode_target(train_df[target_col])
    groups = train_df[group_col].to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    categorical_feature_indices = [feature_cols.index(col) for col in categorical_cols]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, task_name)
        fold_scores: list[float] = []

        for fold, (tr_idx, va_idx) in enumerate(splitter.split(x_train, y_encoded, groups), start=1):
            x_tr = x_train.iloc[tr_idx].copy()
            x_va = x_train.iloc[va_idx].copy()
            y_tr_global = y_encoded[tr_idx]
            y_va = y_true[va_idx]
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_tr_global)

            model = build_model(task_name, params, random_state + fold, len(encoding.classes))
            if TASK_MODEL[task_name] == "xgboost":
                y_tr_model, fold_class_indices = build_fold_target(y_tr_global, task_name, "xgboost")
                x_tr_input = to_xgb_device_matrix(x_tr)
                x_va_input = to_xgb_device_matrix(x_va)
                sample_weight_input = to_xgb_device_vector(sample_weight)
                model.fit(x_tr_input, y_tr_model, sample_weight=sample_weight_input)
                raw_val_prob = to_host_numpy(
                    model.predict_proba(x_va_input),
                    context=f"optuna:{task_name}:xgboost:valid_predict",
                )
                val_prob = align_fold_probabilities(raw_val_prob, len(encoding.classes), fold_class_indices)
                cleanup_gpu_memory()
            else:
                y_tr_model, _ = build_fold_target(y_tr_global, task_name, "catboost")
                model.fit(x_tr, y_tr_model, sample_weight=sample_weight)
                raw_val_prob = to_host_numpy(
                    model.predict_proba(x_va),
                    context=f"optuna:{task_name}:catboost:valid_predict",
                )
                val_prob = align_probabilities(model, raw_val_prob, len(encoding.classes))

            fold_score = score_predictions(y_va, val_prob, metric_name, encoding.classes)
            fold_scores.append(fold_score)
            trial.report(float(np.mean(fold_scores)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


def plot_trials(study_df: pd.DataFrame, metric_name: str, path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(study_df["number"], study_df["value"], marker="o", color="#4C78A8")
    plt.xlabel("trial")
    plt.ylabel(metric_name)
    plt.title(f"Optuna trial history ({metric_name})")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_feature_engineering:
        print("[OPTUNA] feature engineering from raw data")
        build_feature_sets(args.raw_train_path, args.raw_test_path, args.feature_dir)
        args.train_path = args.feature_dir / "train_transition_features.csv"
        args.test_path = args.feature_dir / "test_rally_features.csv"
        args.catalog_path = args.feature_dir / "feature_catalog.csv"

    feature_cols = load_feature_columns(args.catalog_path)
    train_df, test_df = load_frames(args.train_path, args.test_path, feature_cols, args.group_col)
    x_train, x_test, categorical_cols = build_encoded_matrices(train_df, test_df, feature_cols)

    summary: dict[str, Any] = {
        "n_trials": args.n_trials,
        "n_splits": args.n_splits,
        "group_col": args.group_col,
        "tasks": {},
    }

    for task_name in args.tasks:
        task_dir = args.out_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{task_name}_{TASK_MODEL[task_name]}",
            storage=f"sqlite:///{(task_dir / 'study.db').as_posix()}",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
            sampler=optuna.samplers.TPESampler(seed=args.random_state),
        )

        objective = objective_factory(
            task_name=task_name,
            train_df=train_df,
            x_train=x_train,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            group_col=args.group_col,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

        best = {
            "model": TASK_MODEL[task_name],
            "best_value": float(study.best_value),
            "best_params": study.best_params,
            "n_trials_total": len(study.trials),
        }
        (task_dir / "best_params.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")

        study_df = study.trials_dataframe()
        study_df.to_csv(task_dir / "trials.csv", index=False, encoding="utf-8-sig")
        plot_trials(study_df.dropna(subset=["value"]), TASKS[task_name]["metric"], task_dir / "trial_history.png")
        summary["tasks"][task_name] = best

    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
