from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_OUT_DIR = ROOT / "features"

RAW_CONTEXT_COLUMNS = [
    "rally_uid",
    "sex",
    "match",
    "numberGame",
    "rally_id",
    "strikeNumber",
    "scoreSelf",
    "scoreOther",
    "gamePlayerId",
    "gamePlayerOtherId",
    "strikeId",
    "handId",
    "strengthId",
    "spinId",
    "pointId",
    "actionId",
    "positionId",
]

LEAKY_COLUMNS = {"serverGetPoint"}

ACTION_GROUP = {
    0: "Zero",
    1: "Attack",
    2: "Attack",
    3: "Attack",
    4: "Attack",
    5: "Attack",
    6: "Attack",
    7: "Attack",
    8: "Control",
    9: "Control",
    10: "Control",
    11: "Control",
    12: "Defense",
    13: "Defense",
    14: "Defense",
    15: "Serve",
    16: "Serve",
    17: "Serve",
    18: "Serve",
}


def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby("rally_uid", sort=False)

    for col in ["actionId", "pointId", "spinId", "positionId", "handId", "strengthId", "strikeId"]:
        df[f"prev_{col}"] = grp[col].shift(1).fillna(-1).astype(int)
        df[f"next_{col}"] = grp[col].shift(-1)

    df["has_next"] = grp["strikeNumber"].shift(-1).notna()
    df["target_serverGetPoint"] = df["serverGetPoint"]

    df["score_diff"] = df["scoreSelf"] - df["scoreOther"]
    df["score_total"] = df["scoreSelf"] + df["scoreOther"]
    df["lead_state"] = np.sign(df["score_diff"]).astype(int)
    df["is_close_score"] = (df["score_diff"].abs() <= 1).astype(int)
    df["is_deuce_like"] = ((df["scoreSelf"] >= 10) & (df["scoreOther"] >= 10)).astype(int)
    df["is_game_point_self"] = (
        (df["scoreSelf"] >= 10) & (df["scoreSelf"] > df["scoreOther"])
    ).astype(int)
    df["is_game_point_other"] = (
        (df["scoreOther"] >= 10) & (df["scoreOther"] > df["scoreSelf"])
    ).astype(int)

    df["strike_parity"] = (df["strikeNumber"] % 2).astype(int)
    df["serve_receive_phase"] = np.select(
        [df["strikeId"].eq(1), df["strikeId"].eq(2)],
        ["serve", "receive"],
        default="rally",
    )
    df["stage_bucket"] = pd.cut(
        df["strikeNumber"],
        bins=[0, 1, 2, 4, 8, 100],
        labels=["1_serve", "2_receive", "3_4_early", "5_8_mid", "9plus_late"],
        right=True,
    ).astype(str)
    df["action_group"] = df["actionId"].map(ACTION_GROUP).fillna("Unknown")

    df["action_spin_combo"] = df["actionId"].astype(str) + "|" + df["spinId"].astype(str)
    df["action_point_combo"] = df["actionId"].astype(str) + "|" + df["pointId"].astype(str)
    df["hand_action_combo"] = df["handId"].astype(str) + "|" + df["actionId"].astype(str)
    df["action_bigram"] = df["prev_actionId"].astype(str) + "->" + df["actionId"].astype(str)
    df["spin_bigram"] = df["prev_spinId"].astype(str) + "->" + df["spinId"].astype(str)

    df["action_changed"] = (df["actionId"] != df["prev_actionId"]).astype(int)
    df["point_changed"] = (df["pointId"] != df["prev_pointId"]).astype(int)
    df["spin_changed"] = (df["spinId"] != df["prev_spinId"]).astype(int)
    df["position_changed"] = (df["positionId"] != df["prev_positionId"]).astype(int)
    df["hand_changed"] = (df["handId"] != df["prev_handId"]).astype(int)

    action_group = df["action_group"]
    df["prefix_attack_count"] = action_group.eq("Attack").groupby(df["rally_uid"]).cumsum()
    df["prefix_control_count"] = action_group.eq("Control").groupby(df["rally_uid"]).cumsum()
    df["prefix_defense_count"] = action_group.eq("Defense").groupby(df["rally_uid"]).cumsum()
    df["prefix_serve_count"] = action_group.eq("Serve").groupby(df["rally_uid"]).cumsum()
    df["prefix_unique_action_count"] = (
        df.groupby("rally_uid")["actionId"].expanding().nunique().reset_index(level=0, drop=True)
    ).astype(int)

    same_action_run_len: list[int] = []
    for _, group in df.groupby("rally_uid", sort=False):
        prev_value = None
        streak = 0
        for value in group["actionId"]:
            if value == prev_value:
                streak += 1
            else:
                streak = 1
                prev_value = value
            same_action_run_len.append(streak)
    df["same_action_run_len"] = same_action_run_len
    df["same_action_run_bucket"] = pd.cut(
        df["same_action_run_len"],
        bins=[0, 1, 2, 3, 100],
        labels=["1", "2", "3", "4plus"],
        right=True,
    ).astype(str)

    return df


def build_transition_train(train: pd.DataFrame) -> pd.DataFrame:
    df = train[train["has_next"]].copy()
    df["next_actionId"] = df["next_actionId"].astype(int)
    df["next_pointId"] = df["next_pointId"].astype(int)
    return df


def build_rally_test_view(test: pd.DataFrame) -> pd.DataFrame:
    return test.groupby("rally_uid", sort=False).tail(1).copy().reset_index(drop=True)


def build_train_last_view(train: pd.DataFrame) -> pd.DataFrame:
    return train.groupby("rally_uid", sort=False).tail(1).copy().reset_index(drop=True)


def feature_catalog(df: pd.DataFrame) -> pd.DataFrame:
    target_columns = {"next_actionId", "next_pointId", "target_serverGetPoint"}
    identifier_columns = {"rally_uid", "match", "rally_id", "gamePlayerId", "gamePlayerOtherId"}
    recommended_features = {
        "sex",
        "numberGame",
        "strikeNumber",
        "scoreSelf",
        "scoreOther",
        "score_diff",
        "score_total",
        "lead_state",
        "is_close_score",
        "is_deuce_like",
        "is_game_point_self",
        "is_game_point_other",
        "strikeId",
        "handId",
        "strengthId",
        "spinId",
        "pointId",
        "actionId",
        "positionId",
        "strike_parity",
        "serve_receive_phase",
        "stage_bucket",
        "action_group",
        "action_spin_combo",
        "action_point_combo",
        "hand_action_combo",
        "prev_actionId",
        "prev_pointId",
        "prev_spinId",
        "prev_positionId",
        "prev_handId",
        "prev_strengthId",
        "prev_strikeId",
        "action_changed",
        "point_changed",
        "spin_changed",
        "position_changed",
        "hand_changed",
        "action_bigram",
        "spin_bigram",
        "prefix_attack_count",
        "prefix_control_count",
        "prefix_defense_count",
        "prefix_serve_count",
        "prefix_unique_action_count",
        "same_action_run_len",
        "same_action_run_bucket",
    }

    rows = []
    for column in df.columns:
        if column in target_columns:
            role = "target"
        elif column in LEAKY_COLUMNS:
            role = "leaky"
        elif column in identifier_columns:
            role = "identifier"
        elif column in recommended_features:
            role = "feature"
        else:
            role = "auxiliary"
        rows.append(
            {
                "column": column,
                "role": role,
                "dtype": str(df[column].dtype),
                "nunique": int(df[column].nunique(dropna=False)),
                "missing_rate": float(df[column].isna().mean()),
                "recommended_for_model": int(role == "feature"),
            }
        )
    return pd.DataFrame(rows)


def build_feature_sets(train_path: Path, test_path: Path, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    train = add_sequence_features(load_frame(train_path))
    test = add_sequence_features(load_frame(test_path))

    train_transition = build_transition_train(train)
    train_last = build_train_last_view(train)
    test_rally = build_rally_test_view(test)

    train_transition.to_csv(out_dir / "train_transition_features.csv", index=False, encoding="utf-8-sig")
    train_last.to_csv(out_dir / "train_rally_last_features.csv", index=False, encoding="utf-8-sig")
    test_rally.to_csv(out_dir / "test_rally_features.csv", index=False, encoding="utf-8-sig")

    catalog = feature_catalog(train_transition)
    catalog.to_csv(out_dir / "feature_catalog.csv", index=False, encoding="utf-8-sig")

    summary = {
        "train_transition_rows": int(len(train_transition)),
        "train_rally_last_rows": int(len(train_last)),
        "test_rally_rows": int(len(test_rally)),
        "feature_columns": catalog.loc[catalog["role"] == "feature", "column"].tolist(),
        "target_columns": ["next_actionId", "next_pointId", "target_serverGetPoint"],
        "identifier_columns": catalog.loc[catalog["role"] == "identifier", "column"].tolist(),
        "leaky_columns": catalog.loc[catalog["role"] == "leaky", "column"].tolist(),
    }
    (out_dir / "feature_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = """# Feature Builder Outputs

## Files

- `train_transition_features.csv`: 每筆為第 t 拍，用來預測第 t+1 拍與 rally 結果。
- `train_rally_last_features.csv`: 每個 `rally_uid` 的最後一筆已觀測狀態，方便做 rally-level 驗證或分布對照。
- `test_rally_features.csv`: 每個 `rally_uid` 的最後一筆已觀測狀態，對齊提交時的推論輸入。
- `feature_catalog.csv`: 每個欄位的角色、型別、缺失率與是否建議入模。
- `feature_summary.json`: 輸出資料集摘要與推薦特徵清單。

## Targets

- `next_actionId`
- `next_pointId`
- `target_serverGetPoint`

## Warning

- `serverGetPoint` 是現成標籤，不可作為任務三特徵使用。
- `match`、`rally_id`、`gamePlayerId`、`gamePlayerOtherId` 屬識別欄位，建議只在有適當編碼策略時使用。
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    return {
        "out_dir": str(out_dir),
        "train_transition": str(out_dir / "train_transition_features.csv"),
        "train_rally_last": str(out_dir / "train_rally_last_features.csv"),
        "test_rally": str(out_dir / "test_rally_features.csv"),
        "catalog": str(out_dir / "feature_catalog.csv"),
        "summary": str(out_dir / "feature_summary.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reusable feature tables for AI-CUP table-tennis sequences.")
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train.csv")
    parser.add_argument("--test", type=Path, default=DATA_DIR / "test.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_feature_sets(args.train, args.test, args.out_dir)
    print(json.dumps(paths, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
