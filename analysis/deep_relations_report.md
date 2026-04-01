# 深度欄位關聯與特徵分析

## 分析框架

本次改用 transition 視角：對每一個 rally 的第 t 拍，分析當前欄位對第 t+1 拍 `actionId`、`pointId` 與 rally 結果 `serverGetPoint` 的關係。

- 轉換後可用樣本數：`69,712` 筆 transition。
- `serverGetPoint` 在 rally 內是否固定：`100.0%`。
- `test.csv` 的平均已觀測長度：`2.90` 拍。

## 每個原始欄位與目標欄位的關係

- `sex`: 對 `next_actionId` 最強（Cramer's V=0.245）；其次為 `next_pointId`（0.175）；欄位角色屬於 `context`。全局背景欄位，跨比賽穩定，主要提供賽事族群差異。
- `match`: 對 `next_actionId` 最強（Cramer's V=0.253）；其次為 `next_pointId`（0.204）；欄位角色屬於 `identifier`。高基數識別欄位，可能帶場次偏差，適合做分組驗證，不建議直接當主特徵。
- `numberGame`: 對 `next_actionId` 最強（Cramer's V=0.045）；其次為 `next_pointId`（0.030）；欄位角色屬於 `context`。局數上下文，可反映比賽進程與壓力階段。
- `rally_id`: 對 `target_serverGetPoint` 最強（Cramer's V=0.060）；其次為 `next_actionId`（0.028）；欄位角色屬於 `identifier`。局內 rally 編號，偏識別用途，可能有順序偏差。
- `strikeNumber`: 對 `next_actionId` 最強（Cramer's V=0.165）；其次為 `target_serverGetPoint`（0.125）；欄位角色屬於 `temporal`。序列位置核心欄位，與回合結果關係通常很強。
- `scoreSelf`: 對 `target_serverGetPoint` 最強（Cramer's V=0.039）；其次為 `next_actionId`（0.022）；欄位角色屬於 `context`。rally 開始前主視角分數，適合搭配差分與壓力特徵。
- `scoreOther`: 對 `target_serverGetPoint` 最強（Cramer's V=0.035）；其次為 `next_actionId`（0.021）；欄位角色屬於 `context`。rally 開始前對手分數，適合搭配差分與壓力特徵。
- `serverGetPoint`: 對 `target_serverGetPoint` 最強（Cramer's V=1.000）；其次為 `next_pointId`（0.033）；欄位角色屬於 `stroke_state`。若直接用於預測回合勝負會形成標籤洩漏，需謹慎。
- `gamePlayerId`: 對 `next_actionId` 最強（Cramer's V=0.227）；其次為 `next_pointId`（0.175）；欄位角色屬於 `identifier`。球員 ID，高基數，較適合做 embedding 或對戰統計。
- `gamePlayerOtherId`: 對 `next_actionId` 最強（Cramer's V=0.280）；其次為 `next_pointId`（0.179）；欄位角色屬於 `identifier`。對手 ID，高基數，較適合做對戰層級特徵。
- `strikeId`: 對 `next_actionId` 最強（Cramer's V=0.427）；其次為 `next_pointId`（0.173）；欄位角色屬於 `stroke_state`。擊球階段欄位，對下一拍型態與回合結果都很有訊號。
- `handId`: 對 `next_actionId` 最強（Cramer's V=0.285）；其次為 `next_pointId`（0.130）；欄位角色屬於 `stroke_state`。正反手可補技術習慣，但單獨訊號中等。
- `strengthId`: 對 `next_actionId` 最強（Cramer's V=0.272）；其次為 `next_pointId`（0.151）；欄位角色屬於 `stroke_state`。力量層級對下一拍技術有中度關聯。
- `spinId`: 對 `next_actionId` 最強（Cramer's V=0.409）；其次為 `next_pointId`（0.168）；欄位角色屬於 `stroke_state`。旋轉對下一拍 action 很重要。
- `pointId`: 對 `next_actionId` 最強（Cramer's V=0.268）；其次為 `next_pointId`（0.161）；欄位角色屬於 `stroke_state`。落點是重要當前狀態，對下一拍落點/技術有延續性。
- `actionId`: 對 `next_actionId` 最強（Cramer's V=0.278）；其次為 `next_pointId`（0.138）；欄位角色屬於 `stroke_state`。最關鍵當前技術欄位之一，適合做 bigram/轉移特徵。
- `positionId`: 對 `next_actionId` 最強（Cramer's V=0.236）；其次為 `next_pointId`（0.082）；欄位角色屬於 `stroke_state`。站位資訊常與出手技術共同作用。

## 原始欄位彼此的關係

- `sex <-> match`: Cramer's V=1.000
- `strikeNumber <-> strikeId`: Cramer's V=1.000
- `sex <-> gamePlayerId`: Cramer's V=0.997
- `sex <-> gamePlayerOtherId`: Cramer's V=0.997
- `strikeId <-> actionId`: Cramer's V=0.808
- `match <-> gamePlayerOtherId`: Cramer's V=0.712
- `match <-> gamePlayerId`: Cramer's V=0.707
- `spinId <-> actionId`: Cramer's V=0.627
- `gamePlayerId <-> gamePlayerOtherId`: Cramer's V=0.624
- `strikeId <-> positionId`: Cramer's V=0.613
- `rally_id <-> scoreSelf`: Cramer's V=0.588
- `rally_id <-> scoreOther`: Cramer's V=0.587
- `handId <-> actionId`: Cramer's V=0.575
- `strikeId <-> spinId`: Cramer's V=0.533
- `strikeNumber <-> positionId`: Cramer's V=0.500

## 可考慮新增的衍生特徵

- `action_group`: 對 `next_actionId` 的最大關聯度為 0.458，train/test shift TVD=0.129
- `serve_receive_phase`: 對 `next_actionId` 的最大關聯度為 0.427，train/test shift TVD=0.174
- `prev_handId`: 對 `next_actionId` 的最大關聯度為 0.425，train/test shift TVD=0.129
- `position_changed`: 對 `next_actionId` 的最大關聯度為 0.384，train/test shift TVD=0.139
- `prev_strikeId`: 對 `next_actionId` 的最大關聯度為 0.356，train/test shift TVD=0.174
- `action_bigram`: 對 `next_actionId` 的最大關聯度為 0.356，train/test shift TVD=0.220
- `action_point_combo`: 對 `next_actionId` 的最大關聯度為 0.329，train/test shift TVD=0.193
- `stage_bucket`: 對 `next_actionId` 的最大關聯度為 0.319，train/test shift TVD=0.174
- `prev_spinId`: 對 `next_actionId` 的最大關聯度為 0.310，train/test shift TVD=0.140
- `prev_positionId`: 對 `next_actionId` 的最大關聯度為 0.299，train/test shift TVD=0.203
- `prev_strengthId`: 對 `next_actionId` 的最大關聯度為 0.297，train/test shift TVD=0.129
- `action_spin_combo`: 對 `next_actionId` 的最大關聯度為 0.294，train/test shift TVD=0.193
- `hand_action_combo`: 對 `next_actionId` 的最大關聯度為 0.287，train/test shift TVD=0.171
- `prev_actionId`: 對 `next_actionId` 的最大關聯度為 0.269，train/test shift TVD=0.194
- `spin_bigram`: 對 `next_actionId` 的最大關聯度為 0.248，train/test shift TVD=0.148

## 建議保留與避免

- 建議保留：`strikeNumber`、`strikeId`、`actionId`、`pointId`、`spinId`、`positionId`、`strengthId`、`scoreSelf`、`scoreOther`、`score_diff`、`stage_bucket`、`prev_actionId`、`prev_pointId`、`action_bigram`、`prefix_unique_action_count`。
- 高風險或需謹慎：`serverGetPoint` 會直接洩漏回合結果；`match`、`rally_id`、`gamePlayerId`、`gamePlayerOtherId` 屬高基數識別欄位，較適合做 embedding、target encoding 或對戰歷史聚合，不建議裸用。
- 對 `pointId` 任務要保守解讀：雖然 transition 視角比 rally 末端合理，但仍需依主辦方最終切點定義驗證。

## 產出檔案

- `analysis/deep_relations/tables/field_profile.csv`
- `analysis/deep_relations/tables/raw_feature_target_association_transition.csv`
- `analysis/deep_relations/tables/raw_feature_pair_association.csv`
- `analysis/deep_relations/tables/candidate_feature_target_association.csv`
- `analysis/deep_relations/tables/candidate_feature_shortlist.csv`
- `analysis/deep_relations/plots/*.png`