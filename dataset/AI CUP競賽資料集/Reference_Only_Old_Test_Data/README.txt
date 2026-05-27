本資料集（test.csv）為先前之舊版測試資料，因存在 Data Leakage 問題，故不作為本次競賽正式評測使用；正式評測請使用 test_new.csv。

此外，舊版資料（test.csv）中包含 serverGetPoint 欄位，參賽者可自行決定是否使用該資訊進行研究或模型設計；

惟請注意，過度依賴此類洩漏資訊可能導致模型產生過擬合（Overfitting）情形，進而降低模型於 Private 測試資料上的泛化能力與預測表現。