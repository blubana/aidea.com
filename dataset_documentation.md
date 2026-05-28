# 桌球比賽資料集說明文件

## 檔案清單

| 檔案名稱 | 說明 |
|---|---|
| `train.csv` | 訓練資料集，包含完整標註 |
| `test.csv` | 測試資料集，不含預測目標 |
| `sample_submission.csv` | 提交結果格式範例 |

---

## 資料結構與欄位說明

每一筆資料代表**一場比賽中某一小分內的單次揮拍行為**，依時間順序排列。

> 參賽隊伍需要檢視每個 ID 的參考資料，選擇適合的嵌入方法，並選取對訓練重要的特徵。

### 基本欄位

| 欄位名稱 | 說明 | Definition |
|---|---|---|
| `rally_uid` | 小分的唯一識別碼 | Unique ID for each rally |
| `sex` | 比賽性別（男=1 / 女=2） | Gender category of the match |
| `match` | 比賽的唯一識別碼 | Unique ID of the match |
| `numberGame` | 小局數（第幾局） | Game (set) number within the match |
| `rally_id` | 小局內的小分編號 | Rally ID within the game |
| `strikeNumber` | 小分內的揮拍次序 | Stroke number within the rally |
| `scoreSelf` | 主視角選手的得分 | Points won by the main-view player |
| `scoreOther` | 對側選手的得分 | Points won by the opponent player |
| `serverGetPoint` | 發球者是否得分（1=是, 0=否） | Whether the server won the point |
| `gamePlayerId` | 主視角選手的 ID | ID of the main-view player |
| `gamePlayerOtherId` | 對側視角選手的 ID | ID of the opponent player |

### 預測目標欄位（標紅欄位）

| 欄位名稱 | 說明 | Definition |
|---|---|---|
| `strikeId` | 揮拍狀態或動作類型 | Stroke action type or state identifier |
| `handId` | 正手或反手揮拍 | Forehand or backhand stroke indicator |
| `strengthId` | 擊球力道 | Stroke strength level |
| `spinId` | 球的旋轉方式 | Type of spin applied to the ball |
| `pointId` | 球的落點位置 | Landing position of the ball on the table |
| `actionId` | 擊球方式 | Stroke or action type |
| `positionId` | 球員站位區域 | Player's court position |

---

## 類別欄位編碼

### `strikeId` — 揮拍狀態

| ID | 說明 | Definition |
|---|---|---|
| 1 | 發球 | Serving |
| 2 | 接發球 | Reserve |
| 4 | 第三板之後 | Rally |
| 8 | 無（未錄影） | Zero |
| 16 | 暫停 | Stop |

---

### `handId` — 正反手

> `0` 代表「無」、「無動作」、「無法判斷狀態」或「其他」

| ID | 說明 | Definition |
|---|---|---|
| 0 | 無 | Zero |
| 1 | 正拍 | Forehand |
| 2 | 反拍 | Backhand |

---

### `strengthId` — 擊球力道

> `0` 代表「無」、「無動作」、「無法判斷狀態」或「其他」

| ID | 說明 | Definition |
|---|---|---|
| 0 | 無 | Zero |
| 1 | 強 | Strong |
| 2 | 中 | Medium |
| 3 | 弱 | Slow |

---

### `spinId` — 旋轉方式

> `0` 代表「無」、「無動作」、「無法判斷狀態」或「其他」

| ID | 說明 | Definition |
|---|---|---|
| 0 | 無 | Zero |
| 1 | 上旋 | Top Spin |
| 2 | 下旋 | Back Spin |
| 3 | 不旋 | No Spin |
| 4 | 側上旋 | Side Top Spin |
| 5 | 側下旋 | Side Back Spin |

---

### `pointId` — 落點位置

> 落點位置以**接球方（player）的慣用手**為基準定義，從「選手的正手／反手」視角標註，非單純依場地左右座標區分。
>
> `0` 代表「無」或「未落在九宮格的位置」（如掛網出界或直接出界）

| ID | 說明 | Definition |
|---|---|---|
| 0 | 無 | Zero |
| 1 | 正手位置短球 | Forehand position near net |
| 2 | 中間短球 | Middle position near net |
| 3 | 反手位置短球 | Backhand position near net |
| 4 | 正手位置半出台球 | Forehand position half-long |
| 5 | 中路半出台球 | Middle position half-long |
| 6 | 反手位置短半出台球 | Backhand position half-long |
| 7 | 正手位置長球 | Forehand position long |
| 8 | 中間長球 | Middle position long |
| 9 | 反手位置長球 | Backhand position long |

---

### `actionId` — 擊球方式

> `0` 代表「無」或「其他」（無法判斷之球種）

| ID | 說明 | Definition | Action Type |
|---|---|---|---|
| 0 | 無 | Zero | — |
| 1 | 拉球 | Drive | Attack |
| 2 | 反拉 | Counter Drive | Attack |
| 3 | 殺球 | Smash | Attack |
| 4 | 擰球 | Backhand Twist | Attack |
| 5 | 快帶 | Fast Drive | Attack |
| 6 | 推擠 | Fast Push | Attack |
| 7 | 挑撥 | Flip | Attack |
| 8 | 拱球 | Pimple's Long Push | Control |
| 9 | 磕球 | Pimple's Fast Push | Control |
| 10 | 搓球 | Long Push | Control |
| 11 | 擺短 | Drop Shot | Control |
| 12 | 削球 | Chop | Defensive |
| 13 | 擋球 | Block | Defensive |
| 14 | 放高球 | Lob | Defensive |
| 15 | 傳統 | Traditional | Serve |
| 16 | 勾手 | Hook | Serve |
| 17 | 逆旋轉 | Reverse | Serve |
| 18 | 下蹲式 | Squat | Serve |

---

### `positionId` — 球員站位

| ID | 說明 | Definition |
|---|---|---|
| 0 | 無 | Null |
| 1 | 左 | Left |
| 2 | 中 | Middle |
| 3 | 右 | Right |
