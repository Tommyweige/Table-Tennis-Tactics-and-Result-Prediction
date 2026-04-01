# AI-CUP 桌球序列專案總結與後續策略

## 文件目的

本文件整合目前已完成的 `EDA`、欄位關聯分析、特徵工程設計與資料風險判讀，提供後續機器學習工程師作為建模、驗證、迭代與提交策略的統一參考。

本文件重點不只是描述資料，而是回答以下問題：

1. 這份資料在競賽任務上應該如何正確解讀。
2. 哪些欄位真的有訊號，哪些欄位容易造成誤判或洩漏。
3. 哪些衍生特徵值得保留。
4. 接下來應該先做什麼、後做什麼，避免走錯方向。

## 一、任務與資料解讀

### 1. 正式任務定義

依 `readme.md`，正式評分任務為三項：

- 任務一：預測下一拍 `actionId`
- 任務二：預測下一拍 `pointId`
- 任務三：預測該回合結果 `serverGetPoint`

評分權重為：

- `actionId`: 0.4
- `pointId`: 0.4
- `serverGetPoint`: 0.2

其中前兩項使用 `Macro F1`，第三項使用 `AUC-ROC`。

### 2. 文件不一致風險

`data/data_caption.md` 中存在兩個需要特別警惕的地方：

- 文字曾提到 `strikeId` 是預測目標
- 文字又提到 `test.csv` 不包含預測目標

但實際上：

- `sample_submission.csv` 只要求輸出 `rally_uid,actionId,pointId,serverGetPoint`
- `test.csv` 仍然含有每拍歷史欄位，例如 `actionId`、`pointId`、`serverGetPoint`

因此目前最合理、也最一致的資料解讀方式是：

- `test.csv` 提供的是每個 `rally_uid` 的已觀測前綴序列
- 需要預測的是下一拍的 `actionId`、`pointId` 與該 rally 結果 `serverGetPoint`

### 3. 資料粒度

原始資料的粒度是：

- 每一列 = 某個 `rally_uid` 中的單次擊球
- 每個 `rally_uid` = 一段完整 rally

因此這不是一般 tabular 問題，而是明確的序列預測問題。

## 二、資料總覽

### 1. 原始規模

- `train.csv`: 84,707 筆拍次資料
- `test.csv`: 3,589 筆拍次資料
- `train` 中共有 14,995 個 rally
- `test` 中共有 1,236 個 rally

### 2. 序列長度特徵

- 訓練資料中平均完整 rally 長度約 `5.65` 拍
- 測試資料中平均已觀測前綴長度約 `2.90` 拍

這代表：

- train 與 test 在序列位置上不完全同分布
- test 比較偏早期 prefix
- 若模型只擅長處理 rally 後段，實戰表現可能明顯下降

### 3. `serverGetPoint` 的性質

分析結果顯示：

- `serverGetPoint` 在同一個 rally 內是固定值

這代表它是天然的 rally-level target，而不是每拍獨立 target。因此：

- 任務三不能逐列隨機切分驗證
- 若把同一 rally 的不同行放進 train/valid，會產生嚴重洩漏

## 三、EDA 核心結論

### 1. 類別不均衡非常明顯

在 rally-level 近似觀察下，`actionId` 類別高度不平衡，最常見類別約為：

- `13:Block` 約 21.2%
- `1:Drive` 約 18.2%

尾端類別極少，例如：

- `16:HookServe` 幾乎不存在

這與競賽採用 `Macro F1` 完全一致，也表示：

- 不能只追求整體 accuracy
- 需要對少數類別額外照顧

### 2. 單純用「最後一拍」近似目標有侷限

若直接把 train 中每個 rally 的最後一拍當作 `pointId` 目標，幾乎全部都會落在 `0`。這說明：

- test 的 prefix 切點很可能不是固定在最後一拍前
- 若直接把問題簡化成「預測最後一拍」，會誤導 `pointId` 任務

因此後續更可靠的做法是使用 transition 視角：

- 用第 `t` 拍資訊預測第 `t+1` 拍

## 四、Transition 視角的深度分析

### 1. 可用樣本數

將 train 轉換成 transition 後，共得到：

- `69,712` 筆訓練樣本

每一筆代表：

- 輸入：第 `t` 拍的狀態與 prefix 特徵
- 輸出：第 `t+1` 拍的 `actionId`、`pointId`，以及 rally 結果 `serverGetPoint`

### 2. `next_actionId` 的主要驅動欄位

對 `next_actionId` 最有訊號的原始欄位依序包含：

- `strikeId`，Cramer's V 約 `0.427`
- `spinId`，約 `0.409`
- `handId`，約 `0.285`
- `actionId`，約 `0.278`
- `strengthId`，約 `0.272`
- `pointId`，約 `0.268`
- `positionId`，約 `0.236`

這代表下一拍技術，主要受當前拍的：

- 擊球階段
- 旋轉
- 正反手
- 當前技術
- 力道
- 落點
- 站位

共同影響。

### 3. `next_pointId` 的主要驅動欄位

對 `next_pointId` 有穩定訊號的欄位包含：

- `strikeId`
- `spinId`
- `pointId`
- `strengthId`
- `actionId`
- `handId`

整體強度雖低於 `next_actionId`，但仍表示：

- 落點具有前後延續性
- 技術型態和旋轉也會共同影響下一拍落點

### 4. `serverGetPoint` 的主要驅動欄位

若排除直接洩漏的 `serverGetPoint` 本欄，對任務三較有意義的訊號主要來自：

- `strikeNumber`
- `scoreSelf`
- `scoreOther`
- `score_diff`
- 一些 prefix 累積特徵

這表示任務三更像是：

- 比分上下文 + 回合長度/階段 + 某些序列動態

而不是單看某一拍的技術欄位。

## 五、欄位彼此之間的關係

幾組特別強的欄位關係如下：

- `strikeNumber <-> strikeId = 1.000`
- `strikeId <-> actionId = 0.808`
- `spinId <-> actionId = 0.627`
- `strikeId <-> positionId = 0.613`
- `handId <-> actionId = 0.575`
- `strikeId <-> spinId = 0.533`

這些結果說明：

1. 欄位之間有很強的結構性，不是彼此獨立。
2. 某些欄位有部分重複資訊，模型可能會學到高度共線的模式。
3. 線性模型或樹模型可直接利用這些關係，但若要壓縮特徵、做 embedding、做序列模型，需留意冗餘。

## 六、欄位風險判讀

### 1. 明確洩漏欄位

- `serverGetPoint`

此欄位對任務三就是現成標籤，因此：

- 不能作為 `target_serverGetPoint` 的輸入特徵
- 若誤用，local validation 會虛高

### 2. 高基數識別欄位

- `match`
- `rally_id`
- `gamePlayerId`
- `gamePlayerOtherId`

這些欄位在統計上看起來可能和目標有關，但很可能只是：

- 場次偏差
- 選手特徵記憶
- 資料集內特定配對效應

因此不建議直接裸用。比較合理的使用方式是：

- embedding
- target encoding
- player history aggregation
- matchup aggregation

### 3. Train/Test Shift 風險

比較大的分布位移主要出現在：

- `strikeId`
- `actionId`
- `positionId`
- `spinId`
- `strengthId`

表示 train 與 test 在序列上下文與技術狀態上存在差異，因此：

- local CV 不能太樂觀
- 模型需要對 prefix 位置與比賽階段更穩健

## 七、值得保留的特徵

### 1. 原始特徵

最建議直接保留的原始欄位：

- `sex`
- `numberGame`
- `strikeNumber`
- `scoreSelf`
- `scoreOther`
- `strikeId`
- `handId`
- `strengthId`
- `spinId`
- `pointId`
- `actionId`
- `positionId`

### 2. 上下文衍生特徵

- `score_diff`
- `score_total`
- `lead_state`
- `is_close_score`
- `is_deuce_like`
- `is_game_point_self`
- `is_game_point_other`

這些特徵對任務三尤其重要，也有助於模型理解比賽壓力狀態。

### 3. 時序與階段特徵

- `strike_parity`
- `serve_receive_phase`
- `stage_bucket`

這些特徵能明確告訴模型目前在：

- 發球階段
- 接發階段
- 前段對抗
- 中後段 rally

### 4. 前一拍與轉移特徵

- `prev_actionId`
- `prev_pointId`
- `prev_spinId`
- `prev_positionId`
- `prev_handId`
- `prev_strengthId`
- `prev_strikeId`
- `action_bigram`
- `spin_bigram`

這些特徵對 `next_actionId` 和 `next_pointId` 幫助很大，因為桌球本質上就是有明顯轉移規律的序列問題。

### 5. 組合特徵

- `action_group`
- `action_spin_combo`
- `action_point_combo`
- `hand_action_combo`

這些特徵提供單欄位沒有的交互資訊，尤其適合樹模型、類別 embedding 模型、或多塔輸入架構。

### 6. 前綴累積與穩定度特徵

- `action_changed`
- `point_changed`
- `spin_changed`
- `position_changed`
- `hand_changed`
- `prefix_attack_count`
- `prefix_control_count`
- `prefix_defense_count`
- `prefix_serve_count`
- `prefix_unique_action_count`
- `same_action_run_len`
- `same_action_run_bucket`

這些特徵反映的是：

- 序列是否穩定
- 技術是否開始轉換
- 前綴是否偏攻擊、偏控制、偏防守

這對後續 sequence-aware 模型或樹模型都有價值。

## 八、目前已完成的特徵建構

目前已實作 `feature_builder.py`，並輸出：

- `features/train_transition_features.csv`
- `features/train_rally_last_features.csv`
- `features/test_rally_features.csv`
- `features/feature_catalog.csv`
- `features/feature_summary.json`

其中：

- `train_transition_features.csv` 是目前最推薦的 baseline 訓練表
- `feature_catalog.csv` 已標註哪些欄位是 `feature`、`identifier`、`leaky`、`target`

目前建議直接作為模型輸入的特徵數量已足夠建立第一版 baseline。

## 九、接下來的策略

### 策略總原則

建議不要一開始就做太複雜的深度模型，而是按以下順序推進：

1. 先建立可靠的 validation 與 baseline
2. 再確認 feature set 是否真的有效
3. 最後才上 sequence model 或更複雜的多任務模型

這樣可以避免：

- 問題定義錯誤
- validation 泄漏
- 模型太強但方向錯誤

## 十、建議執行路線

### 第一階段：建立可信 baseline

目標：

- 快速得到第一個可比較的可重現分數

建議做法：

- 使用 `features/train_transition_features.csv`
- 先建立三個任務分開訓練的 baseline
- 驗證使用 `GroupKFold` by `rally_uid`
- 額外做一版 by `match` 的保守驗證

模型建議：

- `LightGBM`
- `CatBoost`
- `XGBoost`

原因：

- 能快速驗證特徵是否有效
- 對混合型數值/類別特徵較友善
- 容易做 feature importance 與錯誤分析

### 第二階段：處理類別不均衡

重點任務：

- `actionId`
- `pointId`

建議做法：

- class weight
- focal loss
- rare class oversampling
- head/tail two-stage strategy

其中 two-stage 的想法是：

- 先判斷大類或熱門類
- 再在尾端類別中細分

### 第三階段：加強 `pointId` 任務

`pointId` 是目前最需要小心的任務，原因是：

- rally-level 最後一拍解讀會失真
- transition 視角雖合理，但仍未完全保證與競賽真實切點一致

因此建議：

- 先在 transition 表上建立 baseline
- 針對 `pointId` 做單獨錯誤分析
- 檢查不同 `strikeId`、`stage_bucket`、`serve_receive_phase` 下的預測表現

如果 `pointId` 表現持續不佳，可考慮：

- 以 `actionId` 預測結果作為輔助特徵
- 建立多任務學習模型，讓 `actionId` 幫助 `pointId`

### 第四階段：加入多任務或序列模型

當 baseline 與驗證流程穩定後，再考慮：

- `Transformer`
- `RNN/LSTM/GRU`
- `Temporal CNN`
- Tabular + Sequence hybrid model

推薦原因：

- 前一拍到下一拍有明顯 transition structure
- prefix 長度與比賽階段有重要訊號
- 單筆 tabular 會損失部分序列上下文

但啟動時機應該在 baseline 穩定之後，而不是一開始就上。

### 第五階段：處理選手與對戰資訊

如果 baseline 已穩定，可再考慮使用：

- `gamePlayerId`
- `gamePlayerOtherId`
- `match`

但建議方式不是直接 one-hot，而是：

- player embedding
- historical player statistics
- matchup prior features

例如：

- 某球員在 `serve_receive_phase=serve` 時常見 `actionId` 分布
- 某對戰組合在 `spinId` 或 `action_group` 上的偏好

這一層通常能帶來增益，但也最容易 overfit。

## 十一、驗證與提交流程建議

### 1. 驗證切分

至少準備兩套：

- `GroupKFold` by `rally_uid`
- `GroupKFold` by `match`

用途：

- `rally_uid` 驗證比較接近一般資料切分
- `match` 驗證比較能看泛化能力

### 2. 指標設計

本地評估請對齊競賽：

- `actionId`: Macro F1
- `pointId`: Macro F1
- `serverGetPoint`: AUC
- Overall Score = `0.4 * action + 0.4 * point + 0.2 * server`

### 3. 提交策略

建議不要只看單模：

- 先保留最穩定的單模
- 再考慮 cross-validation ensemble
- 最後再考慮任務間互相輔助的融合

## 十二、目前最推薦的下一步

若以實作優先順序來看，最推薦的下一步是：

1. 建立 `baseline_train.py`
2. 以 `features/train_transition_features.csv` 為訓練資料
3. 先做三個任務獨立模型
4. 對齊競賽評分公式輸出 local score
5. 檢查 `actionId`、`pointId` 的少數類別表現
6. 再決定是否加入多任務學習或更完整序列模型

## 十三、結論

這個專案的核心不是單純分類，而是：

- 一個帶有序列轉移規律的多任務預測問題
- 具有明顯類別不均衡
- 帶有 train/test context shift
- 並且存在文件描述不完全一致的風險

目前最合理的做法已經明確：

- 以 transition 視角建立 baseline
- 嚴格避免驗證洩漏
- 保留高價值原始欄位與序列衍生特徵
- 先用樹模型建立可靠基準
- 再逐步升級到序列模型與更進階的 player/matchup 特徵

若後續執行順序正確，這份資料是有機會透過良好特徵工程與穩健驗證，先做出高品質 baseline，再進一步疊代到有競爭力的模型表現。
