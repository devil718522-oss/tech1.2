import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta

st.set_page_config(page_title="家銘趨勢雷達 V1.4.2", page_icon="📡", layout="wide")
st.title("📡 家銘趨勢雷達 V1.4.2")
st.caption("技術面型態雷達｜品質分 × 進場分 × 追高風險 × 歷史驗證")

API = "https://api.finmindtrade.com/api/v4/data"

POOLS = {
    "活潑／趨勢股": ["2330","2317","2454","2303","2382","3231","2379","3034","3711","2409","3481","2344","2603","2609","2002","1301","1303","2881","2882","2886"],
    "大型權值股": ["2330","2317","2454","2303","2382","2881","2882","2891","2886","2412","1301","1303","2002","1216","3711"],
    "電子趨勢觀察": ["2376","2357","3231","2382","2454","3034","3017","2344","2409","2303","2379","3711","6669","3443","3661"],
}

def api_get(dataset, stock_id, start_date, end_date, token=""):
    params = {
        "dataset": dataset,
        "data_id": stock_id,
        "start_date": str(start_date),
        "end_date": str(end_date),
    }
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(API, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    js = r.json()
    if js.get("status") not in (None, 200):
        raise RuntimeError(js.get("msg", "FinMind API error"))
    return pd.DataFrame(js.get("data", []))

@st.cache_data(ttl=1800, show_spinner=False)
def load_price(stock_id, start_date, end_date, token=""):
    d = api_get("TaiwanStockPrice", stock_id, start_date, end_date, token)
    if d.empty:
        return d
    d["date"] = pd.to_datetime(d["date"])
    rename = {
        "open": "Open", "max": "High", "min": "Low", "close": "Close",
        "Trading_Volume": "Volume", "Trading_money": "Money"
    }
    d = d.rename(columns=rename).sort_values("date").drop_duplicates("date")
    for c in ["Open","High","Low","Close","Volume","Money"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["Close"]).reset_index(drop=True)

def features(d):
    x = d.copy()
    for n in [5,10,20,60,120,200]:
        x[f"MA{n}"] = x["Close"].rolling(n).mean()
    x["VMA20"] = x["Volume"].rolling(20).mean()
    x["VR"] = x["Volume"] / x["VMA20"]
    x["R20"] = x["Close"].pct_change(20)
    x["R60"] = x["Close"].pct_change(60)

    # 關鍵：shift(1)，今天的突破只能比較「昨天以前」的20日高點，避免偷看今天。
    x["PrevHH20"] = x["High"].shift(1).rolling(20).max()
    x["PrevHH60"] = x["High"].shift(1).rolling(60).max()
    x["PrevLL20"] = x["Low"].shift(1).rolling(20).min()

    delta = x["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["RSI"] = 100 - 100/(1+rs)

    e12 = x["Close"].ewm(span=12, adjust=False).mean()
    e26 = x["Close"].ewm(span=26, adjust=False).mean()
    x["MACD"] = e12-e26
    x["SIGNAL"] = x["MACD"].ewm(span=9, adjust=False).mean()
    return x

def score_row(x, i=-1):
    r = x.iloc[i]
    pos = len(x)+i if i < 0 else i
    if pos < 205 or pd.isna(r["MA200"]):
        return None

    q = 0
    q_reasons = []

    checks = [
        (r.Close > r.MA20, 10, "股價站上20MA"),
        (r.MA5 > r.MA10 > r.MA20, 12, "5>10>20 多頭排列"),
        (r.MA20 > r.MA60, 10, "20MA>60MA"),
        (r.MA60 > r.MA120, 8, "60MA>120MA"),
        (r.MA120 > r.MA200, 6, "長期結構偏多"),
        (r.MA20 > x["MA20"].iloc[pos-5], 8, "20MA上彎"),
        (r.MA60 > x["MA60"].iloc[pos-10], 6, "60MA上彎"),
        (r.MACD > r.SIGNAL, 6, "MACD偏多"),
        (r.R20 > 0, 5, "20日動能正"),
        (r.R60 > 0.05, 5, "60日動能強"),
        (r.Close >= r.PrevHH20*0.97, 8, "接近20日壓力／高點"),
        (r.VR >= 1.2, 6, "量能放大"),
    ]
    for ok, pts, msg in checks:
        if pd.notna(ok) and bool(ok):
            q += pts
            q_reasons.append(msg)
    q = int(np.clip(q, 0, 100))

    g5 = r.Close/r.MA5 - 1
    g20 = r.Close/r.MA20 - 1
    breakout20 = pd.notna(r.PrevHH20) and r.Close > r.PrevHH20 and r.VR >= 1.25
    breakout60 = pd.notna(r.PrevHH60) and r.Close > r.PrevHH60 and r.VR >= 1.25
    near_ma = (-0.015 <= g5 <= 0.025) or (-0.02 <= g20 <= 0.025)
    healthy_rsi = pd.notna(r.RSI) and 48 <= r.RSI <= 70

    e = 35
    e_reasons = []
    risk = []

    for ok, pts, msg in [
        (r.Close > r.MA20, 10, "守在20MA上"),
        (r.MA5 > r.MA10 > r.MA20, 10, "短均線多排"),
        (r.MACD > r.SIGNAL, 6, "MACD偏多"),
        (healthy_rsi, 8, "RSI健康"),
        (near_ma, 12, "乖離適中／接近均線"),
        (1.0 <= r.VR <= 2.5, 7, "量能健康"),
        (breakout20, 10, "帶量突破20日壓力"),
        (breakout60, 7, "帶量突破60日壓力"),
        (q >= 70, 5, "品質分達標"),
    ]:
        if bool(ok):
            e += pts
            e_reasons.append(msg)

    if g5 > 0.08:
        e -= 35; risk.append(f"5MA乖離過大 {g5:.1%}")
    elif g5 > 0.05:
        e -= 18; risk.append(f"5MA乖離偏高 {g5:.1%}")
    if pd.notna(r.RSI) and r.RSI > 78:
        e -= 30; risk.append(f"RSI過熱 {r.RSI:.0f}")
    elif pd.notna(r.RSI) and r.RSI > 72:
        e -= 12; risk.append(f"RSI偏熱 {r.RSI:.0f}")
    if pd.notna(r.R20) and r.R20 > 0.30:
        e -= 25; risk.append(f"20日漲幅過大 {r.R20:.1%}")
    elif pd.notna(r.R20) and r.R20 > 0.20:
        e -= 12; risk.append(f"20日漲幅偏大 {r.R20:.1%}")
    if r.Close < r.MA20:
        e -= 35; risk.append("跌破20MA")
    e = int(np.clip(e, 0, 100))

    if risk and (g5 > .08 or (pd.notna(r.RSI) and r.RSI > 78)):
        state = "🔴 過熱不追"
    elif breakout20 and e >= 70:
        state = "🟢 突破候選"
    elif q >= 65 and e >= 72 and near_ma:
        state = "🟢 回測／均線買點候選"
    elif q >= 65 and e >= 50:
        state = "🟡 等待買點"
    else:
        state = "⚪ 暫不列入"

    defense = min(r.MA10, r.MA20) if pd.notna(r.MA10) and pd.notna(r.MA20) else np.nan
    return {
        "品質分": q, "進場分": e, "狀態": state,
        "收盤": r.Close, "量比": r.VR, "RSI": r.RSI,
        "5MA乖離": g5, "20日漲跌": r.R20,
        "20日壓力": r.PrevHH20, "參考防守": defense,
        "品質理由": q_reasons, "進場理由": e_reasons, "風險": risk,
    }


def swing_points(x, left=3, right=3, lookback=140):
    z = x.tail(lookback).copy().reset_index()
    hs, ls = [], []
    for j in range(left, len(z)-right):
        if z.loc[j,"High"] >= z.loc[j-left:j+right,"High"].max():
            hs.append((int(z.loc[j,"index"]), float(z.loc[j,"High"])))
        if z.loc[j,"Low"] <= z.loc[j-left:j+right,"Low"].min():
            ls.append((int(z.loc[j,"index"]), float(z.loc[j,"Low"])))
    return hs, ls

def line_value(p1, p2, at_i):
    i1,v1=p1; i2,v2=p2
    if i2==i1: return np.nan
    return v1 + (v2-v1)/(i2-i1)*(at_i-i1)

def technical_structure(x):
    r = x.iloc[-1]
    idx = len(x) - 1
    hs, ls = swing_points(x)
    trend = "盤整"
    if len(hs) >= 2 and len(ls) >= 2:
        if hs[-1][1] > hs[-2][1] and ls[-1][1] > ls[-2][1]:
            trend = "多頭"
        elif hs[-1][1] < hs[-2][1] and ls[-1][1] < ls[-2][1]:
            trend = "空頭"

    up_line = np.nan
    down_line = np.nan
    if len(ls) >= 2 and ls[-1][1] > ls[-2][1]:
        up_line = line_value(ls[-2], ls[-1], idx)
    if len(hs) >= 2 and hs[-1][1] < hs[-2][1]:
        down_line = line_value(hs[-2], hs[-1], idx)

    # ABC：最近 A低-B高-C低，多頭修正結構
    abc = False
    A = B = C = np.nan
    if len(ls) >= 2 and len(hs) >= 1:
        candidates = []
        for a in ls:
            for b in hs:
                for c in ls:
                    if a[0] < b[0] < c[0]:
                        candidates.append((a, b, c))
        if candidates:
            a, b, c = max(candidates, key=lambda t: t[2][0])
            A, B, C = a[1], b[1], c[1]
            abc = (B > A and C > A and C < B)

    # 壓力：以「已知的前高」為主。即使現價已突破，也保留原突破基準，才能判斷是否追高。
    resistance_candidates = [v for v in [r.PrevHH20, hs[-1][1] if hs else np.nan, B] if pd.notna(v)]
    below_or_near = [v for v in resistance_candidates if v <= r.Close * 1.03]
    resistance = max(below_or_near) if below_or_near else (min(resistance_candidates) if resistance_candidates else np.nan)
    right_trigger = float(resistance) * 1.002 if pd.notna(resistance) else np.nan

    # 回測區與結構防守
    pull_vals = [v for v in [r.MA10, r.MA20, up_line] if pd.notna(v)]
    pullback_zone_low = min(pull_vals) if pull_vals else np.nan
    pullback_zone_high = max(pull_vals) if pull_vals else np.nan

    stop_candidates = [v for v in [r.MA20, up_line, C, ls[-1][1] if ls else np.nan]
                       if pd.notna(v) and v < r.Close]
    structural_stop = max(stop_candidates) if stop_candidates else float(r.Close) * 0.95
    # 留一點價格雜訊空間，避免剛碰支撐就被洗掉
    structural_stop *= 0.995

    # 左側觸發改用「前一交易日高點」，今日收盤才有可能真正觸發；不再用今日 High 自己追自己。
    prev_high = float(x["High"].iloc[-2]) if len(x) >= 2 else np.nan
    touched_pullback = (trend == "多頭" and pd.notna(pullback_zone_high)
                        and r.Low <= pullback_zone_high * 1.02
                        and r.Close >= pullback_zone_low * 0.98)
    left_trigger = prev_high * 1.002 if touched_pullback and pd.notna(prev_high) else np.nan

    # 型態目標
    target = np.nan
    target_type = "—"
    if abc:
        target = C + (B - A)
        target_type = "ABC等幅"
    elif pd.notna(r.PrevHH20) and pd.notna(r.PrevLL20):
        target = r.PrevHH20 + (r.PrevHH20 - r.PrevLL20)
        target_type = "20日箱型等幅"

    # 技術共振
    resonance = 0
    reasons = []
    for ok, msg in [
        (trend == "多頭", "頭頭高、底底高"),
        (r.MA5 > r.MA10 > r.MA20, "5>10>20多排"),
        (r.MA20 > r.MA60, "20MA>60MA"),
        (pd.notna(up_line) and r.Close >= up_line, "守上升切線"),
        (abc, "ABC多頭修正"),
        (r.VR >= 1.0, "量能不弱"),
        (r.MACD > r.SIGNAL, "MACD偏多"),
        (r.Close > r.MA20, "站上20MA"),
    ]:
        if bool(ok):
            resonance += 1
            reasons.append(msg)

    # === V1.4.1 交易決策層 ===
    close = float(r.Close)
    atr = (x["High"] - x["Low"]).rolling(14).mean().iloc[-1]
    atr_pct = float(atr / close) if pd.notna(atr) and close > 0 else 0.02
    chase_limit = max(0.025, min(0.06, atr_pct * 1.5))  # 個股波動越大，容許區略寬

    action = "⚪ 結構未完成"
    entry_low = entry_high = planned_entry = np.nan
    trigger_type = "—"
    breakout_gap = np.nan

    if close < structural_stop:
        action = "🔴 結構失效／不進"
    elif pd.notna(right_trigger) and close >= right_trigger:
        breakout_gap = close / right_trigger - 1
        trigger_type = "右側突破"
        if breakout_gap <= chase_limit:
            # 已突破但仍在合理區：不叫使用者回頭買舊突破價，改用目前可接受區間
            entry_low = max(right_trigger, close * 0.985)
            entry_high = min(close * 1.01, right_trigger * (1 + chase_limit))
            planned_entry = (entry_low + entry_high) / 2
            action = "🟢 突破後合理區"
        else:
            # 已離突破點太遠，等待回測，不追價
            entry_low = right_trigger * 0.995
            entry_high = right_trigger * 1.015
            planned_entry = (entry_low + entry_high) / 2
            action = "🟡 已突破／乖離過大，等回測"
    elif pd.notna(left_trigger) and close >= left_trigger:
        trigger_type = "左側回測轉強"
        entry_low = left_trigger
        entry_high = left_trigger * 1.015
        planned_entry = (entry_low + entry_high) / 2
        action = "🟢 回測轉強可觀察"
    elif trend == "多頭" and pd.notna(right_trigger):
        trigger_type = "等待右側突破"
        entry_low = right_trigger
        entry_high = right_trigger * 1.015
        planned_entry = (entry_low + entry_high) / 2
        action = "🟡 等待突破"
    elif trend == "多頭" and pd.notna(left_trigger):
        trigger_type = "等待左側轉強"
        entry_low = left_trigger
        entry_high = left_trigger * 1.015
        planned_entry = (entry_low + entry_high) / 2
        action = "🟡 等待回測轉強"

    # 第一目標：至少先看 1.5R；型態目標作第二目標。
    rr = np.nan
    target1 = np.nan
    if pd.notna(planned_entry) and planned_entry > structural_stop:
        risk_per_share = planned_entry - structural_stop
        target1 = planned_entry + risk_per_share * 1.5
        if pd.notna(target) and target > planned_entry:
            rr = (target - planned_entry) / risk_per_share

    # 風報比過濾：不足 1.5R 不值得做；若型態目標已在現價下方，也直接放棄。
    if pd.notna(planned_entry):
        if pd.notna(target) and target <= planned_entry:
            action = "🔴 目標空間不足／不進"
        elif pd.notna(rr) and rr < 1.5:
            action = "🔴 風報比不足／不進"

    return {
        "趨勢結構": trend, "共振": resonance, "共振理由": reasons,
        "上升切線": up_line, "下降切線": down_line,
        "ABC": abc, "A": A, "B": B, "C": C,
        "回測區下": pullback_zone_low, "回測區上": pullback_zone_high,
        "左側觸發": left_trigger, "右側突破": right_trigger,
        "進場區下": entry_low, "進場區上": entry_high, "預定進場": planned_entry,
        "觸發類型": trigger_type, "突破乖離": breakout_gap,
        "結構停損": structural_stop, "第一目標": target1,
        "目標價": target, "目標類型": target_type,
        "風報比": rr, "結構狀態": action, "追價容許": chase_limit
    }


def final_trade_decision(s, ts):
    """把分數與結構整合成一個最終燈號；不是勝率或報酬保證。"""
    quality = float(s["品質分"])
    entry_score = float(s["進場分"])
    resonance = int(ts["共振"])
    vr = float(s["量比"]) if pd.notna(s["量比"]) else 0.0
    rr = ts["風報比"]
    state = ts["結構狀態"]

    failed = []
    if quality < 65:
        failed.append(f"品質 {quality:.0f}<65")
    if entry_score < 72:
        failed.append(f"進場分 {entry_score:.0f}<72")
    if resonance < 6:
        failed.append(f"共振 {resonance}/8<6/8")
    if vr < 1.0:
        failed.append(f"量比 {vr:.2f}<1.00")
    if pd.isna(rr) or rr < 1.5:
        failed.append("風報比未達 1:1.5")

    zone = "-"
    if pd.notna(ts["進場區下"]) and pd.notna(ts["進場區上"]):
        zone = f'{ts["進場區下"]:.2f}～{ts["進場區上"]:.2f}'

    if "結構失效" in state or "不進" in state:
        return "🔴 不進場", state.replace("🔴 ",""), zone
    if failed:
        return "🔴 不進場", "；".join(failed), zone
    if "乖離過大" in state:
        return "🔵 等待回測", f"已突破但離觸發點過遠；等回測 {zone}", zone
    if "等待突破" in state:
        p = ts["右側突破"]
        reason = f"突破 {p:.2f} 且量比維持≥1.00再評估" if pd.notna(p) else "等待右側突破"
        return "🟡 等待突破", reason, zone
    if "等待回測轉強" in state:
        return "🟡 等待轉強", f"等待回測後重新轉強；觀察 {zone}", zone
    if ("突破後合理區" in state or "回測轉強可觀察" in state):
        return "🟢 可進場候選", f"條件通過；參考進場區 {zone}", zone
    return "⚪ 觀察", state.replace("⚪ ",""), zone


def grade(v):
    return "S" if v >= 85 else "A" if v >= 75 else "B" if v >= 65 else "C"

def backtest(x, threshold=72, cooldown=5, cost=0.00585):
    rows = []
    last_signal = -999
    # 僅用當天及以前資料產生訊號；未來報酬只在訊號產生後才計算。
    for i in range(205, len(x)-21):
        if i-last_signal < cooldown:
            continue
        s = score_row(x, i)
        if not s or s["進場分"] < threshold or s["品質分"] < 65:
            continue
        if s["狀態"] not in ("🟢 突破候選","🟢 回測／均線買點候選"):
            continue
        entry = x["Close"].iloc[i]
        rec = {"date": x["date"].iloc[i], "entry": entry, "品質分": s["品質分"], "進場分": s["進場分"], "型態": s["狀態"]}
        for h in [5,10,20]:
            rec[f"r{h}"] = x["Close"].iloc[i+h]/entry - 1 - cost
        future = x.iloc[i+1:i+21]
        rec["MFE20"] = future["High"].max()/entry - 1
        rec["MAE20"] = future["Low"].min()/entry - 1
        rows.append(rec)
        last_signal = i
    return pd.DataFrame(rows)

with st.sidebar:
    token = st.text_input("FinMind Token（可留空；依 API 權限而定）", type="password")
    history_days = st.slider("下載歷史日數", 450, 1800, 900, 50)
    st.caption("V1.4.2：新增最終交易燈號，把品質、進場分、共振、量價、結構與風報比整合成單一決策。")

mode = st.radio("掃描模式", ["📚 內建選股池", "✍️ 手動檢測"], horizontal=True)

if mode == "📚 內建選股池":
    pool_name = st.selectbox("選擇股票池", list(POOLS.keys()))
    stocks = POOLS[pool_name]
    st.caption(f"目前股票池：{pool_name}｜{len(stocks)} 檔")
    with st.expander("查看本次股票池"):
        st.write("、".join(stocks))
else:
    raw = st.text_area("輸入股票代號｜一行一個", "2376\n3481\n2330", height=180)
    stocks = [s.strip() for s in raw.replace(",", "\n").splitlines() if s.strip().isdigit()]

if st.button("📡 啟動 V1.4.2 技術決策雷達", type="primary", use_container_width=True):
    end = date.today()
    start = end - timedelta(days=history_days)
    rows, details, datasets = [], {}, {}
    bar = st.progress(0, text="準備掃描…")
    for n, sid in enumerate(stocks):
        try:
            p = load_price(sid, start, end, token)
            if len(p) < 220:
                raise RuntimeError(f"歷史資料不足：{len(p)} 筆")
            x = features(p)
            s = score_row(x)
            if not s:
                raise RuntimeError("無法計算完整技術指標")
            ts = technical_structure(x)
            final_light, final_reason, final_zone = final_trade_decision(s, ts)
            rows.append({
                "代號": sid, "品質分": s["品質分"], "級別": grade(s["品質分"]),
                "進場分": s["進場分"], "狀態": s["狀態"], "收盤": s["收盤"],
                "量比": s["量比"], "RSI": s["RSI"], "5MA乖離": s["5MA乖離"],
                "20日漲跌": s["20日漲跌"], "趨勢結構": ts["趨勢結構"],
                "技術共振": ts["共振"], "結構狀態": ts["結構狀態"],
                "左側觸發": ts["左側觸發"], "右側突破": ts["右側突破"],
                "進場區下": ts["進場區下"], "進場區上": ts["進場區上"],
                "結構停損": ts["結構停損"], "第一目標": ts["第一目標"],
                "目標價": ts["目標價"], "風報比": ts["風報比"],
                "最終決策": final_light, "決策原因": final_reason
            })
            s["structure"] = ts
            details[sid] = s
            datasets[sid] = x
        except Exception as e:
            details[sid] = {"error": str(e)}
        bar.progress((n+1)/max(len(stocks),1), text=f"掃描 {n+1}/{len(stocks)}｜{sid}")
    bar.empty()

    if not rows:
        st.error("沒有成功取得可分析資料。請檢查 FinMind API/Token、股票代號或稍後再試。")
    else:
        out = pd.DataFrame(rows).sort_values(["進場分","品質分"], ascending=False)
        show = out.copy()
        show["量比"] = show["量比"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
        show["RSI"] = show["RSI"].map(lambda v: f"{v:.0f}" if pd.notna(v) else "-")
        show["5MA乖離"] = show["5MA乖離"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "-")
        show["20日漲跌"] = show["20日漲跌"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "-")

        st.session_state["radar_out"] = out
        st.session_state["radar_show"] = show
        st.session_state["radar_details"] = details
        st.session_state["radar_datasets"] = datasets

if "radar_out" in st.session_state:
    out = st.session_state["radar_out"]
    show = st.session_state["radar_show"]
    details = st.session_state["radar_details"]
    datasets = st.session_state["radar_datasets"]

    st.header("🚦 今日最終交易燈號")
    st.caption("先看這裡：品質分、進場分、技術共振、量價、結構與風報比必須一起通過，才會列為綠燈。")

    final_cols = ["代號","最終決策","決策原因","收盤","進場區下","進場區上","結構停損","第一目標","目標價","風報比"]
    final_view = out[final_cols].copy()
    priority = {"🟢 可進場候選":0, "🟡 等待突破":1, "🟡 等待轉強":2, "🔵 等待回測":3, "⚪ 觀察":4, "🔴 不進場":5}
    final_view["_p"] = final_view["最終決策"].map(priority).fillna(9)
    final_view = final_view.sort_values(["_p","風報比"], ascending=[True,False]).drop(columns="_p")
    for c in ["收盤","進場區下","進場區上","結構停損","第一目標","目標價"]:
        final_view[c] = final_view[c].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
    final_view["風報比"] = final_view["風報比"].map(lambda v: "-" if pd.isna(v) else f"1:{v:.2f}")
    st.dataframe(final_view, use_container_width=True, hide_index=True)

    green = out[out["最終決策"]=="🟢 可進場候選"]
    if green.empty:
        st.info("今天沒有綠燈候選。沒有符合條件時就等待，不為了交易而交易。")
    else:
        st.success(f"目前有 {len(green)} 檔綠燈候選；仍需以實際成交量與當下 K 線確認。")

    st.header("🏆 股票品質 × 進場排行榜")
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.header("🧭 技術結構決策")
    decision_cols=["代號","趨勢結構","技術共振","結構狀態","收盤","右側突破","進場區下","進場區上","結構停損","第一目標","目標價","風報比"]
    decision=out[decision_cols].copy()
    for c in ["收盤","右側突破","進場區下","進場區上","結構停損","第一目標","目標價"]:
        decision[c]=decision[c].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
    decision["風報比"]=decision["風報比"].map(lambda v:"-" if pd.isna(v) else f"1:{v:.2f}")
    st.dataframe(decision,use_container_width=True,hide_index=True)

    st.subheader("📐 個股結構詳解")
    for _,dr in out.sort_values(["技術共振","進場分"],ascending=False).head(8).iterrows():
        sid=dr["代號"]; ts=details[sid]["structure"]
        with st.expander(f'{sid}｜共振 {ts["共振"]}/8｜{ts["結構狀態"]}'):
            st.write("**技術共振：** "+("、".join(ts["共振理由"]) or "尚未形成"))
            if ts["ABC"]:
                st.write(f'**ABC：** A {ts["A"]:.2f} → B {ts["B"]:.2f} → C {ts["C"]:.2f}')
            c1,c2,c3,c4=st.columns(4)
            c1.metric("右側突破","-" if pd.isna(ts["右側突破"]) else f'{ts["右側突破"]:.2f}')
            zone = "-" if pd.isna(ts["進場區下"]) else f'{ts["進場區下"]:.2f}～{ts["進場區上"]:.2f}'
            c2.metric("建議進場區", zone)
            c3.metric("跌破退出","-" if pd.isna(ts["結構停損"]) else f'{ts["結構停損"]:.2f}')
            c4.metric("第一目標","-" if pd.isna(ts["第一目標"]) else f'{ts["第一目標"]:.2f}')
            st.write(f'**型態目標（{ts["目標類型"]}）：** '+("-" if pd.isna(ts["目標價"]) else f'{ts["目標價"]:.2f}'))
            st.write(f'**目前決策：{ts["結構狀態"]}**')
            if pd.notna(ts["回測區下"]) and pd.notna(ts["回測區上"]):
                st.caption(f'回測觀察區：{ts["回測區下"]:.2f} ～ {ts["回測區上"]:.2f}')
            if pd.notna(ts["風報比"]):
                st.write(f'**預估風報比：1 : {ts["風報比"]:.2f}**')
            st.caption("V1.4.2 會區分『突破觸發價』與『目前可接受進場區』；已離突破點過遠會改為等待回測。價位仍為規則化研究值，需搭配成交量與實際K線確認。")

    st.header("🎯 現在的進場候選")
    candidates = out[out["最終決策"].isin(["🟢 可進場候選","🟡 等待突破","🟡 等待轉強","🔵 等待回測"])].copy()
    if candidates.empty:
        st.info("目前沒有通過條件的候選。雷達不強迫產生買點。")
    else:
        for _, row in candidates.head(10).iterrows():
            sid = row["代號"]
            s = details[sid]
            with st.expander(f'{sid}｜品質 {s["品質分"]}/100｜進場 {s["進場分"]}/100｜{s["狀態"]}'):
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("收盤", f'{s["收盤"]:.2f}')
                c2.metric("量比", f'{s["量比"]:.2f}')
                c3.metric("20日壓力", f'{s["20日壓力"]:.2f}' if pd.notna(s["20日壓力"]) else "-")
                c4.metric("參考防守", f'{s["參考防守"]:.2f}' if pd.notna(s["參考防守"]) else "-")
                st.write("**品質成立：** " + ("、".join(s["品質理由"]) or "無"))
                st.write("**進場加分：** " + ("、".join(s["進場理由"]) or "無"))
                if s["風險"]:
                    st.warning("追高／結構風險：" + "、".join(s["風險"]))

    st.divider()
    st.header("🧪 歷史驗證｜不偷看未來")
    st.caption("只以訊號當日以前資料決定是否觸發；再觀察其後 5／10／20 交易日結果。")
    c1,c2,c3 = st.columns(3)
    bt_threshold = c1.slider("最低進場分", 60, 90, 72, 2)
    bt_cooldown = c2.slider("同股訊號冷卻（日）", 1, 20, 5)
    bt_cost = c3.number_input("單筆來回成本（%）", 0.0, 2.0, 0.585, 0.05)/100

    bt_ids = st.multiselect("驗證股票", list(datasets.keys()), default=list(datasets.keys())[:min(10,len(datasets))])
    if st.button("開始歷史驗證", use_container_width=True):
        all_bt = []
        for sid in bt_ids:
            b = backtest(datasets[sid], bt_threshold, bt_cooldown, bt_cost)
            if not b.empty:
                b.insert(0, "代號", sid)
                all_bt.append(b)
        if not all_bt:
            st.warning("這組門檻沒有產生足夠歷史訊號，可降低最低進場分再測。")
        else:
            bt = pd.concat(all_bt, ignore_index=True)
            st.session_state["bt"] = bt

    if "bt" in st.session_state:
        bt = st.session_state["bt"]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("訊號數", len(bt))
        c2.metric("20日勝率", f'{(bt["r20"]>0).mean():.1%}')
        c3.metric("20日平均", f'{bt["r20"].mean():.2%}')
        wins = bt.loc[bt["r20"]>0,"r20"].sum()
        losses = -bt.loc[bt["r20"]<0,"r20"].sum()
        pf = wins/losses if losses > 0 else np.nan
        c4.metric("Profit Factor", f"{pf:.2f}" if pd.notna(pf) else "∞")

        summary = pd.DataFrame({
            "期間":["5日","10日","20日"],
            "勝率":[(bt["r5"]>0).mean(),(bt["r10"]>0).mean(),(bt["r20"]>0).mean()],
            "平均報酬":[bt["r5"].mean(),bt["r10"].mean(),bt["r20"].mean()],
            "中位數報酬":[bt["r5"].median(),bt["r10"].median(),bt["r20"].median()],
        })
        st.dataframe(summary.style.format({"勝率":"{:.1%}","平均報酬":"{:.2%}","中位數報酬":"{:.2%}"}), use_container_width=True, hide_index=True)
        st.caption(f'20日平均 MFE：{bt["MFE20"].mean():.2%}｜平均 MAE：{bt["MAE20"].mean():.2%}')
        st.download_button("下載歷史驗證 CSV", bt.to_csv(index=False).encode("utf-8-sig"), "v1_3_backtest.csv", "text/csv")

st.caption("V1.4.2｜研究與策略驗證用途，不構成投資建議。參考防守為技術結構提示，不是個人化停損建議。")
