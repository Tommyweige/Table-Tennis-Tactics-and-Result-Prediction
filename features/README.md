# Feature Builder Outputs

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
