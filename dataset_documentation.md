# 桌球比賽小分資料集 — 完整說明文件

## 資料集簡介

以「桌球比賽小分（Rally）」為單位所建立的結構化比賽資料集，完整記錄比賽中每一次擊球的狀態、技術動作與比賽脈絡資訊。

在測試資料集（`test.csv`）中，參賽者須根據每一個小分（`rally_uid`）的前 n-1 擊球資訊，預測：

1. **第 n 拍的球種**（`actionId`）
2. **第 n 拍的落點**（`pointId`）
3. **該小分的最終勝負**（`serverGetPoint`）

> 第 n 拍並非固定為回合結束拍次，勝負可能發生在第 n 拍或後續第 n+1、n+2… 拍。

---

## 檔案說明

| 檔案名稱 | 說明 |
|---|---|
| `train.csv` | 訓練資料集，包含完整標註 |
| `test.csv` | 測試資料集，不含預測目標 |
| `sample_submission.csv` | 提交結果格式範例 |

---

## 資料結構

每一筆資料代表**一場比賽中某一小分內的單次揮拍行為**，依時間順序排列。

### 欄位總覽

| 欄位名稱 | 中文說明 | Definition |
|---|---|---|
| `rally_uid` | 小分的唯一識別碼 | Unique ID for each rally |
| `sex` | 比賽性別（男=1 / 女=2） | Gender category of the match |
| `match` | 比賽的唯一識別碼 | Unique ID of the match |
| `numberGame` | 小局數（第幾局） | Game (set) number within the match |
| `rally_id` | 小局內的小分編號 | Rally ID within the game |
| `strikeNumber` | 小分內的揮拍次序 | Stroke number within the rally |
| `scoreSelf` | 主視角選手的得分 | Points won by the main-view player |
| `scoreOther` | 對側選手的得分 | Points won by the opponent player |
| `serverGetPoint` ⭐ | 發球者是否得分（1=是, 0=否） | Whether the server won the point |
| `gamePlayerId` | 主視角選手的 ID | ID of the main-view player |
| `gamePlayerOtherId` | 對側視角選手的 ID | ID of the opponent player |
| `strikeId` | 揮拍狀態或動作類型 | Stroke action type or state identifier |
| `handId` | 正手或反手揮拍 | Forehand or backhand stroke indicator |
| `strengthId` | 擊球力道 | Stroke strength level |
| `spinId` | 球的旋轉方式 | Type of spin applied to the ball |
| `pointId` ⭐ | 球的落點位置 | Landing position of the ball on the table |
| `actionId` ⭐ | 擊球方式 | Stroke or action type |
| `positionId` | 球員站位區域 | Player's court position |

> ⭐ 為本次競賽預測目標欄位

---

## 類別欄位詳細定義

### `strikeId` — 揮拍狀態

| ID | 中文說明 | Definition |
|---|---|---|
| 1 | 發球 | Serving |
| 2 | 接發球 | Reserve |
| 4 | 第三板之後 | Rally |
| 8 | 無（未錄影） | Zero |
| 16 | 暫停 | Stop |

---

### `handId` — 正反手

| ID | 中文說明 | Definition |
|---|---|---|
| 0 | 無 | Zero |
| 1 | 正拍 | Forehand |
| 2 | 反拍 | Backhand |

> 0 代表「無」、「無動作」、「無法判斷狀態」或「其他」

---

### `strengthId` — 擊球力道

| ID | 中文說明 | Definition |
|---|---|---|
| 0 | 無 | Zero |
| 1 | 強 | Strong |
| 2 | 中 | Medium |
| 3 | 弱 | Slow |

> 0 代表「無」、「無動作」、「無法判斷狀態」或「其他」

---

### `spinId` — 旋轉方式

| ID | 中文說明 | Definition |
|---|---|---|
| 0 | 無 | Zero |
| 1 | 上旋 | Top spin |
| 2 | 下旋 | Back spin |
| 3 | 不旋 | No spin |
| 4 | 側上旋 | Side top spin |
| 5 | 側下旋 | Side back spin |

> 0 代表「無」、「無動作」、「無法判斷狀態」或「其他」

---

### `pointId` — 落點位置 ⭐

> 落點以**接球方（player）的慣用手**為基準定義，從「選手的正手／反手」視角標註，而非單純依場地左右座標區分。

| ID | 中文說明 | Definition |
|---|---|---|
| 0 | 無 | Zero（掛網、出界，未落在九宮格） |
| 1 | 正手位置短球 | Forehand position near net |
| 2 | 中間短球 | Middle position near net |
| 3 | 反手位置短球 | Backhand position near net |
| 4 | 正手位置半出台球 | Forehand position half-long |
| 5 | 中路半出台球 | Middle position half-long |
| 6 | 反手位置半出台球 | Backhand position half-long |
| 7 | 正手位置長球 | Forehand position long |
| 8 | 中間長球 | Middle position long |
| 9 | 反手位置長球 | Backhand position long |

> 落點九宮格示意（接球方視角）：
>
> ```
> 正手短(1) | 中短(2) | 反手短(3)
> 正手半(4) | 中半(5) | 反手半(6)
> 正手長(7) | 中長(8) | 反手長(9)
> ```

---

### `actionId` — 擊球方式 ⭐

| ID | 中文說明 | Definition | Action Type |
|---|---|---|---|
| 0 | 無 | Zero | — |
| 1 | 拉球 | Drive | Attack |
| 2 | 反拉 | Counter drive | Attack |
| 3 | 殺球 | Smash | Attack |
| 4 | 擰球 | Backhand twist | Attack |
| 5 | 快帶 | Fast drive | Attack |
| 6 | 推擠 | Fast push | Attack |
| 7 | 挑撥 | Flip | Attack |
| 8 | 拱球 | Pimple's long push | Control |
| 9 | 磕球 | Pimple's fast push | Control |
| 10 | 搓球 | Long push | Control |
| 11 | 擺短 | Drop shot | Control |
| 12 | 削球 | Chop | Defensive |
| 13 | 擋球 | Block | Defensive |
| 14 | 放高球 | Lob | Defensive |
| 15 | 傳統 | Traditional | Serve |
| 16 | 勾手 | Hook | Serve |
| 17 | 逆旋轉 | Reverse | Serve |
| 18 | 下蹲式 | Squat | Serve |

> 0 代表「無」或「其他」（無法判斷之球種）

---

### `positionId` — 球員站位

| ID | 中文說明 | Definition |
|---|---|---|
| 0 | 無 | Null |
| 1 | 左 | Left |
| 2 | 中 | Middle |
| 3 | 右 | Right |

> 0 代表「無」、「無動作」、「無法判斷狀態」或「其他」

---

## 預測目標摘要

| 目標欄位 | 類型 | 類別數 |
|---|---|---|
| `actionId` | 多分類 | 19（0–18） |
| `pointId` | 多分類 | 10（0–9） |
| `serverGetPoint` | 二元分類 | 2（0 / 1） |
