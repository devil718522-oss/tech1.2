import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta

st.set_page_config(page_title="家銘趨勢雷達 V1.4", page_icon="📡", layout="wide")
st.title("📡 家銘趨勢雷達 V1.4")
st.caption("技術面型態雷達｜品質 × 結構 × 價格觸發 × 量價 × 風報比")

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
    r=x.iloc[-1]; idx=len(x)-1
    hs,ls=swing_points(x)
    trend="盤整"
    if len(hs)>=2 and len(ls)>=2:
        if hs[-1][1]>hs[-2][1] and ls[-1][1]>ls[-2][1]: trend="多頭"
        elif hs[-1][1]<hs[-2][1] and ls[-1][1]<ls[-2][1]: trend="空頭"

    up_line=np.nan; down_line=np.nan
    if len(ls)>=2 and ls[-1][1]>ls[-2][1]:
        up_line=line_value(ls[-2],ls[-1],idx)
    if len(hs)>=2 and hs[-1][1]<hs[-2][1]:
        down_line=line_value(hs[-2],hs[-1],idx)

    # ABC：最近三個交替的重要轉折，以 A低-B高-C低 的多頭修正為主
    abc=False; A=B=C=np.nan
    if len(ls)>=2 and len(hs)>=1:
        candidates=[]
        for a in ls:
            for b in hs:
                for c in ls:
                    if a[0] < b[0] < c[0]:
                        candidates.append((a,b,c))
        if candidates:
            a,b,c=max(candidates,key=lambda t:t[2][0])
            A,B,C=a[1],b[1],c[1]
            abc=(B>A and C>A and C<B)

    prev_high = r.PrevHH20
    resistance = min([v for v in [prev_high, hs[-1][1] if hs else np.nan, B] if pd.notna(v) and v>=r.Close],
                     default=prev_high if pd.notna(prev_high) else np.nan)
    breakout_trigger = (float(resistance)*1.002) if pd.notna(resistance) else np.nan

    supports=[v for v in [r.MA10,r.MA20,up_line,C,ls[-1][1] if ls else np.nan] if pd.notna(v) and v<r.Close]
    structural_stop=max(supports) if supports else float(r.Close)*0.95

    # 左側：回測10/20MA或上升切線後，以當日高點+0.2%作確認；右側：突破主要壓力
    pullback_zone_low=min([v for v in [r.MA10,r.MA20,up_line] if pd.notna(v)], default=np.nan)
    pullback_zone_high=max([v for v in [r.MA10,r.MA20,up_line] if pd.notna(v)], default=np.nan)
    prev_bar_high=float(x.iloc[-2].High) if len(x)>=2 else np.nan
    left_trigger=prev_bar_high*1.002 if trend=="多頭" and pd.notna(prev_bar_high) and pd.notna(pullback_zone_high) and r.Low<=pullback_zone_high*1.02 else np.nan
    right_trigger=breakout_trigger

    # 型態目標：ABC 等幅 AB 自 C 投射；否則20日箱體等幅
    target=np.nan; target_type="—"
    if abc:
        target=C+(B-A); target_type="ABC等幅"
    elif pd.notna(r.PrevHH20) and pd.notna(r.PrevLL20):
        target=r.PrevHH20+(r.PrevHH20-r.PrevLL20); target_type="20日箱型等幅"

    # 交易計畫：把「突破價」與「真正預定進場價」分開。
    # 右側突破後，只允許在突破價上方 2% 內視為合理追價區；超過則等回測。
    entry_low=np.nan; entry_high=np.nan; planned_entry=np.nan
    first_target=np.nan; rr=np.nan
    if pd.notna(right_trigger):
        entry_low=float(right_trigger)
        entry_high=float(right_trigger)*1.02
        if r.Close < right_trigger:
            planned_entry=float(right_trigger)
        elif r.Close <= entry_high:
            planned_entry=float(r.Close)
        else:
            planned_entry=float(right_trigger)*1.01

    if pd.notna(planned_entry) and pd.notna(target) and target>planned_entry:
        first_target=planned_entry + (target-planned_entry)*0.5
    if pd.notna(planned_entry) and pd.notna(target) and planned_entry>structural_stop and target>planned_entry:
        rr=(target-planned_entry)/(planned_entry-structural_stop)

    resonance=0; reasons=[]
    for ok,msg in [
        (trend=="多頭","頭頭高、底底高"),
        (r.MA5>r.MA10>r.MA20,"5>10>20多排"),
        (r.MA20>r.MA60,"20MA>60MA"),
        (pd.notna(up_line) and r.Close>=up_line,"守上升切線"),
        (abc,"ABC多頭修正"),
        (r.VR>=1.0,"量能不弱"),
        (r.MACD>r.SIGNAL,"MACD偏多"),
        (r.Close>r.MA20,"站上20MA"),
    ]:
        if bool(ok): resonance+=1; reasons.append(msg)

    if r.Close < structural_stop:
        status="🔴 結構失效"
    elif pd.notna(right_trigger) and r.Close > right_trigger*1.02:
        status="🔵 已突破／等待回測"
    elif pd.notna(right_trigger) and r.Close>=right_trigger:
        status="🟢 突破後合理區"
    elif pd.notna(left_trigger) and r.Close>=left_trigger:
        status="🟢 左側轉強"
    elif trend=="多頭":
        status="🟡 等待突破"
    else:
        status="⚪ 結構未完成"

    return {
        "趨勢結構":trend,"共振":resonance,"共振理由":reasons,
        "上升切線":up_line,"下降切線":down_line,
        "ABC":abc,"A":A,"B":B,"C":C,
        "回測區下":pullback_zone_low,"回測區上":pullback_zone_high,
        "左側觸發":left_trigger,"右側突破":right_trigger,
        "進場區下":entry_low,"進場區上":entry_high,"預定進場":planned_entry,
        "結構停損":structural_stop,"第一目標":first_target,
        "目標價":target,"目標類型":target_type,
        "風報比":rr,"結構狀態":status
    }


def final_trade_decision(s, ts):
    """整合品質、共振、價格位置、量價與風報比，輸出唯一最終決策。"""
    q=float(s["品質分"])
    resonance=int(ts["共振"])
    close=float(s["收盤"])
    vr=float(s["量比"]) if pd.notna(s["量比"]) else 0.0
    rt=ts["右側突破"]
    stop=ts["結構停損"]
    rr=ts["風報比"]

    reasons=[]
    # 先處理硬性排除
    if close <= stop:
        return "🔴 不進場", "結構已失效", reasons
    if q < 65:
        return "🔴 不進場", f"品質 {q:.0f} 分未達 65", reasons
    if resonance < 5:
        return "🔴 不進場", f"技術共振 {resonance}/8 未達 5/8", reasons
    if pd.notna(rr) and rr < 1.5:
        return "🔴 不進場", f"風報比僅 1:{rr:.2f}，低於 1:1.5", reasons

    # 尚未突破：品質/結構夠好才進入等待名單
    if pd.notna(rt) and close < rt:
        if vr < 0.8:
            reasons.append("目前量能偏弱")
        return "🟡 等待突破", f"突破 {rt:.2f} 後再確認", reasons

    # 已突破太遠：不追價
    if pd.notna(rt) and close > rt*1.02:
        return "🔵 等待回測", f"已高於突破價 2%，等回測 {rt:.2f}～{rt*1.02:.2f}", reasons

    # 在合理突破區，仍要求量能不能太弱
    if pd.notna(rt) and rt <= close <= rt*1.02:
        if vr < 0.8:
            return "🟡 等量確認", f"價格已到位，但量比 {vr:.2f} 偏弱", reasons
        return "🟢 可進場", f"合理進場區 {rt:.2f}～{rt*1.02:.2f}", reasons

    return "🟡 觀察", "尚未形成完整價格觸發", reasons


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
    st.caption("V1.4：品質 × 進場 × 技術結構 × 明確觸發/停損/目標價。")

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

if st.button("📡 啟動 V1.4.3 一眼看懂進場版", type="primary", use_container_width=True):
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
            final_state, final_action, final_notes = final_trade_decision(s, ts)
            rows.append({
                "代號": sid, "品質分": s["品質分"], "級別": grade(s["品質分"]),
                "進場分": s["進場分"], "狀態": s["狀態"], "收盤": s["收盤"],
                "量比": s["量比"], "RSI": s["RSI"], "5MA乖離": s["5MA乖離"],
                "20日漲跌": s["20日漲跌"], "趨勢結構": ts["趨勢結構"],
                "技術共振": ts["共振"], "結構狀態": ts["結構狀態"],
                "最終決策": final_state, "現在該做": final_action,
                "左側觸發": ts["左側觸發"], "右側突破": ts["右側突破"],
                "進場區下": ts["進場區下"], "進場區上": ts["進場區上"],
                "結構停損": ts["結構停損"], "第一目標": ts["第一目標"],
                "目標價": ts["目標價"], "風報比": ts["風報比"]
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

    st.header("🚦 現在能不能買？")
    st.caption("先看結論，再看技術細節。只有「🟢 可以進場」才是目前符合完整規則的標的。")

    priority={"🟢 可進場":0,"🟡 等待突破":1,"🟡 等量確認":2,"🔵 等待回測":3,"🟡 觀察":4,"🔴 不進場":5}
    cards=out.copy()
    cards["_p"]=cards["最終決策"].map(priority).fillna(9)
    cards=cards.sort_values(["_p","品質分","技術共振"],ascending=[True,False,False])

    for _,row in cards.head(12).iterrows():
        sid=row["代號"]
        decision=row["最終決策"]

        if decision=="🟢 可進場":
            title=f"🟢 {sid}｜可以進場"
            headline=f'建議進場：{row["進場區下"]:.2f} ～ {row["進場區上"]:.2f} 元'
        elif decision=="🟡 等待突破":
            title=f"🟡 {sid}｜還不能買"
            headline=f'等待突破：{row["右側突破"]:.2f} 元'
        elif decision=="🟡 等量確認":
            title=f"🟡 {sid}｜價格到了，等成交量"
            headline=f'現價 {row["收盤"]:.2f} 元；量比 {row["量比"]:.2f}，量能確認後再評估'
        elif decision=="🔵 等待回測":
            title=f"🔵 {sid}｜不要追價"
            headline=f'等待回測：{row["進場區下"]:.2f} ～ {row["進場區上"]:.2f} 元'
        elif decision=="🔴 不進場":
            title=f"🔴 {sid}｜目前不進場"
            headline=str(row["現在該做"])
        else:
            title=f"🟡 {sid}｜先觀察"
            headline=str(row["現在該做"])

        with st.container(border=True):
            st.subheader(title)
            st.markdown(f"### {headline}")
            a,b,c,d=st.columns(4)
            a.metric("現價",f'{row["收盤"]:.2f}')
            b.metric("停損","-" if pd.isna(row["結構停損"]) else f'{row["結構停損"]:.2f}')
            c.metric("第一目標","-" if pd.isna(row["第一目標"]) else f'{row["第一目標"]:.2f}')
            d.metric("型態目標","-" if pd.isna(row["目標價"]) else f'{row["目標價"]:.2f}')

            if decision=="🔴 不進場":
                st.error(f'不進場原因：{row["現在該做"]}')
            elif decision=="🟢 可進場":
                st.success(
                    f'進場 {row["進場區下"]:.2f}～{row["進場區上"]:.2f}｜'
                    f'停損 {row["結構停損"]:.2f}｜'
                    f'風報比 {"-" if pd.isna(row["風報比"]) else f"1:{row["風報比"]:.2f}"}'
                )
            elif decision=="🟡 等待突破":
                st.warning(f'現在先不買；突破 {row["右側突破"]:.2f} 元後，再看量能是否確認。')
            elif decision=="🔵 等待回測":
                st.info(f'已突破但不追高；回到 {row["進場區下"]:.2f}～{row["進場區上"]:.2f} 再重新評估。')
            elif decision=="🟡 等量確認":
                st.warning("價格已進入合理區，但成交量尚未達條件。")

            st.caption(
                f'品質 {row["品質分"]:.0f}/100｜共振 {row["技術共振"]}/8｜'
                f'量比 {row["量比"]:.2f}｜'
                f'風報比 {"-" if pd.isna(row["風報比"]) else f"1:{row["風報比"]:.2f}"}'
            )

    with st.expander("🔎 查看技術細節（進階）"):
        detail_cols=["代號","最終決策","收盤","品質分","技術共振","量比",
                     "右側突破","進場區下","進場區上","結構停損","第一目標","目標價","風報比"]
        detail=out[detail_cols].copy()
        for c in ["收盤","右側突破","進場區下","進場區上","結構停損","第一目標","目標價"]:
            detail[c]=detail[c].map(lambda v:"-" if pd.isna(v) else f"{v:.2f}")
        detail["量比"]=detail["量比"].map(lambda v:"-" if pd.isna(v) else f"{v:.2f}")
        detail["風報比"]=detail["風報比"].map(lambda v:"-" if pd.isna(v) else f"1:{v:.2f}")
        st.dataframe(detail,use_container_width=True,hide_index=True)
        st.caption("技術結構、品質分與共振用來解釋決策；不要把單一綠色結構訊號視為買進訊號。")

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

st.caption("V1.4｜研究與策略驗證用途，不構成投資建議。參考防守為技術結構提示，不是個人化停損建議。")
