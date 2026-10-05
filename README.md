# 家銘趨勢雷達 V1.3

## 核心
- 品質分：判斷股票趨勢結構是否夠強
- 進場分：判斷目前位置是否適合觀察進場
- 追高風險：5MA乖離、RSI、20日漲幅過熱扣分
- 型態：突破候選／回測均線候選／等待買點／過熱不追
- 股票池：內建選股池 + 手動輸入
- 歷史驗證：5/10/20交易日報酬、勝率、Profit Factor、MFE/MAE
- 防 look-ahead：突破高點使用 shift(1) 後的歷史 rolling high

## Streamlit 部署
Main file path：app.py

## FinMind
在左側欄輸入自己的 FinMind Token。未輸入時能否取得資料取決於 FinMind 當下 API 權限與限制。

## 注意
本工具用於研究與策略驗證，不構成投資建議。
