# 桌球序列資料 EDA 報告

## 分析目的

本報告根據 `readme.md` 與 `data/data_caption.md` 的任務定義，將資料整理成更貼近競賽實際提交格式的 rally-level 視角，提供後續機器學習工程師作為建模、驗證與特徵工程參考。

## 重要解讀

1. 評分目標以 `readme.md` 為主，正式任務是預測 `actionId`、`pointId`、`serverGetPoint`。
2. `sample_submission.csv` 只有每個 `rally_uid` 一列，因此建議把 `train.csv` 轉成「觀測前綴序列 -> 最後一拍與 rally 結果」的監督資料。
3. `test.csv` 雖然欄位上仍有 `actionId`、`pointId` 等歷史資訊，但更合理的解釋是這些是已觀測前綴中的拍次內容，而不是待提交標籤。
4. 不過從訓練資料直接取「最後一拍」當作 `pointId` 目標時，幾乎全部都會落在 `0`，這表示實際競賽的 test 前綴切點很可能不是固定落在最後一拍前；因此本報告把這個 rally-level 轉換視為近似分析框架，而不是唯一正解。

## 核心發現

1. 訓練集共有 84,707 筆拍次資料、14,995 個 rally；測試集共有 3,589 筆資料、1,236 個 rally。
2. 若依提交格式回推成競賽真正建模單位，訓練集中可形成 14,995 個『前綴序列 -> 下一拍/回合結果』樣本；平均完整 rally 長度 5.65 拍，測試中已觀測前綴平均長度 2.90 拍。
3. `serverGetPoint` 在同一個 rally 內是固定值，適合作為 rally-level label；也代表驗證切分絕對不能逐列隨機切，必須至少以 `rally_uid` 分組，最好再評估以 `match` 分組的泛化能力。
4. 以「最後一拍近似目標」的角度看，下一拍 `actionId` 類別高度不平衡，最常見類別是 `13:Block`（21.2%），最少見類別是 `16:HookServe`（0.0%）。但 `pointId` 幾乎全部落在 `0:Zero`，這反而支持 test 前綴切點並非固定在最後一拍前。換句話說，`actionId` 的不均衡可直接拿來設計 loss；`pointId` 則需要額外確認主辦方的切點邏輯。
5. 以前綴最後一拍來看，對下一拍 `actionId` 最有關聯的欄位是 `strikeId`（Cramer's V=0.447）；對 `pointId` 最有關聯的是前一拍 `pointId`（0.080）；對 `serverGetPoint` 最有關聯的是 `strikeNumber`（0.998）。其中 `serverGetPoint` 幾乎被拍次奇偶主導，代表回合長度與出手方順序帶有極強訊號。
6. 最明顯的前一拍 -> 下一拍轉移模式是 `10:LongPush` -> `1:Drive`，在同一前一拍條件下的轉移機率為 57.9%。這說明序列轉移訊號很強，單純把資料當 tabular 可能會浪費時序資訊。
7. 訓練前綴最後一拍與測試最後一拍之間的最大分布差異出現在 `strikeId`（TVD=0.265），其次是 `actionId`（0.264）、`positionId`（0.256）、`spinId`（0.197）。這些欄位都值得在特徵工程與驗證設計時做 shift-aware 檢查。
8. 依前綴長度分群後，`serverGetPoint` 勝率呈現非常明顯的奇偶震盪，表示 rally 長度不只是一般 context feature，而可能近似決定最後得分方。
9. 文件存在兩個需要注意的不一致處： `readme.md` 將正式評分目標定義為 `actionId`、`pointId`、`serverGetPoint`； `data_caption.md` 卻又提到 `strikeId` 是預測目標，且同時說 `test.csv` 不含目標。 實際上 `test.csv` 仍包含每拍的 `actionId`/`pointId`/`serverGetPoint` 歷史欄位， 但 `sample_submission.csv` 只有每個 `rally_uid` 一列，因此較合理的競賽解讀是： `test.csv` 提供的是每個 rally 的已觀測前綴序列，要提交的是下一拍與該 rally 結果。


## 建模建議

1. 驗證集切分至少用 `GroupKFold` by `rally_uid`，若要更保守可再做 by `match` 的外部驗證。
2. 任務一與任務二請優先處理類別不均衡，可考慮 class weight、focal loss、重採樣、或 two-stage tail handling。
3. 特徵工程上，至少保留前綴最後一拍的 `actionId`、`pointId`、`spinId`、`strengthId`、`handId`、`positionId`、`score_diff` 與 `prefix_len`。
4. 若模型允許，優先嘗試 sequence model（Transformer、RNN、Temporal CNN、set/sequence encoder），因為前一拍到下一拍的轉移結構很明顯。
5. train/test 在部分上下文欄位有分布位移，建議監控 leaderboard 與 local CV 落差，必要時加入 shift-robust 特徵或重加權。

## 產出檔案

- `analysis/eda_report.md`
- `analysis/eda_dashboard.html`
- `analysis/eda/plots/*.png`
- `analysis/eda/tables/*.csv`
