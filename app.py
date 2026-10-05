import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta

st.set_page_config(page_title="家銘趨勢雷達 1.2", page_icon="📡", layout="wide")
st.title("📡 家銘趨勢雷達 1.2")
st.caption("技術面型態雷達：趨勢 → 均線 → 量價 → K線觸發 → 風險/停損。不是照搬任何付費軟體，而是把公開的技術分析原則量化。")

API="https://api.finmindtrade.com/api/v4/data"
DEFAULT="""2330
2317
2454
2303
2382
3231
2379
3034
3711
2409
3481
2344
2603
2609
2002
1301
1303
2881
2882
2886"""

def fm(dataset, sid=None, start=None, end=None, token=""):
    q={"dataset":dataset}
    if sid not in (None,""): q["data_id"]=sid
    if start is not None: q["start_date"]=str(start)
    if end is not None: q["end_date"]=str(end)
    h={"Authorization":f"Bearer {token}"} if token else {}
    r=requests.get(API,params=q,headers=h,timeout=30)
    r.raise_for_status()
    js=r.json()
    if js.get("status") not in (None,200):
        raise RuntimeError(js.get("msg","FinMind API error"))
    return pd.DataFrame(js.get("data",[]))

@st.cache_data(ttl=3600,show_spinner=False)
def get_price(sid,start,end,token):
    d=fm("TaiwanStockPrice",sid,start,end,token)
    if d.empty:return d
    d["date"]=pd.to_datetime(d["date"])
    ren={"open":"Open","max":"High","min":"Low","close":"Close","Trading_Volume":"Volume"}
    d=d.rename(columns=ren).sort_values("date").reset_index(drop=True)
    for c in ["Open","High","Low","Close","Volume"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    return d.dropna(subset=["Open","High","Low","Close","Volume"])

def indicators(d):
    x=d.copy()
    for n in [5,10,20,60]:
        x[f"MA{n}"]=x.Close.rolling(n).mean()
    x["VMA20"]=x.Volume.rolling(20).mean()
    x["VOLR"]=x.Volume/x.VMA20
    x["RET3"]=x.Close.pct_change(3)
    x["RET20"]=x.Close.pct_change(20)
    x["HH20"]=x.High.rolling(20).max()
    x["LL20"]=x.Low.rolling(20).min()
    x["BODY"]=(x.Close-x.Open).abs()
    x["RANGE"]=(x.High-x.Low).replace(0,np.nan)
    return x

def pivots(x,left=3,right=3):
    highs=[]; lows=[]
    h=x.High.values; l=x.Low.values
    for i in range(left,len(x)-right):
        if h[i] == np.max(h[i-left:i+right+1]): highs.append((i,h[i]))
        if l[i] == np.min(l[i-left:i+right+1]): lows.append((i,l[i]))
    return highs,lows

def trend6(x):
    hs,ls=pivots(x.tail(100).reset_index(drop=True))
    if len(hs)<2 or len(ls)<2:return "盤整/未確認",None
    h1,h2=hs[-2][1],hs[-1][1]; l1,l2=ls[-2][1],ls[-1][1]
    if h2>h1 and l2>l1:return "多頭",{"前高":h1,"近高":h2,"前低":l1,"近低":l2}
    if h2<h1 and l2<l1:return "空頭",{"前高":h1,"近高":h2,"前低":l1,"近低":l2}
    return "盤整/未確認",{"前高":h1,"近高":h2,"前低":l1,"近低":l2}

def analyze(x):
    r=x.iloc[-1]; prev=x.iloc[-2]
    tr,pv=trend6(x)
    ma3 = r.MA5>r.MA10>r.MA20
    ma4 = ma3 and r.MA20>r.MA60
    above20 = r.Close>r.MA20
    ma20up = r.MA20>x.MA20.iloc[-6]
    red = r.Close>r.Open
    stand5 = red and prev.Close<=prev.MA5 and r.Close>r.MA5
    stand10 = red and prev.Close<=prev.MA10 and r.Close>r.MA10
    stand20 = red and prev.Close<=prev.MA20 and r.Close>r.MA20
    breakout = r.Close>x.High.iloc[-21:-1].max() if len(x)>=21 else False
    volume_ok = r.Volume>=r.VMA20
    vol_breakout = breakout and r.Volume>=1.2*r.VMA20

    # 進貨量：近20日中，量>20均量但單日漲幅沒有失控
    z=x.tail(20).copy()
    pct=z.Close.pct_change().abs()
    accumulation=((z.Volume>z.VMA20)&(pct<0.04)).sum()

    gap20=r.Close/r.MA20-1
    chase = (r.RET3>0.10) or (gap20>0.12)
    danger = (r.Volume>=2*r.VMA20 and r.Close<r.Open and r.Close<prev.Low)

    tags=[]
    if tr=="多頭":tags.append("頭頭高、底底高")
    if ma3:tags.append("5>10>20 多排")
    if ma4:tags.append("5>10>20>60 多排")
    if accumulation>=2:tags.append(f"近20日進貨量 {accumulation} 次")
    if vol_breakout:tags.append("帶量突破")
    if stand5:tags.append("紅K站回5MA")
    if stand10:tags.append("紅K站回10MA")
    if stand20:tags.append("紅K站回20MA")

    # 型態分類：規則優先於總分
    radar=[]
    if tr=="多頭" and ma3 and breakout and volume_ok: radar.append("🚀 突破雷達")
    if tr=="多頭" and ma3 and (stand5 or stand10 or stand20): radar.append("🎯 回檔轉強")
    if accumulation>=2 and above20 and ma20up and breakout: radar.append("🔥 起漲雷達")
    if tr=="多頭" and ma4 and above20: radar.append("💪 強勢多頭")
    if tr=="空頭" or r.Close<r.MA20: radar.append("⚠️ 轉弱/排除")

    # 透明分數只用來排序，不等同買進
    score=0
    score += 25 if tr=="多頭" else 0
    score += 20 if ma3 else 0
    score += 10 if ma4 else 0
    score += 10 if ma20up else 0
    score += min(accumulation,3)*5
    score += 10 if breakout else 0
    score += 5 if volume_ok else 0
    score += 5 if (stand5 or stand10 or stand20) else 0
    score -= 20 if chase else 0
    score -= 25 if danger else 0
    score=int(np.clip(score,0,100))

    if danger: action="🔴 高檔大量長黑風險"
    elif chase: action="🟠 強勢但乖離大，不追"
    elif "🎯 回檔轉強" in radar: action="🟢 回檔轉強候選"
    elif "🚀 突破雷達" in radar: action="🟢 突破候選"
    elif tr=="多頭" and ma3: action="🟡 多頭觀察"
    else: action="⚪ 等待"

    stop_candidates=[v for v in [r.MA10,r.MA20,(pv or {}).get("近低")] if pd.notna(v)]
    stop=max([v for v in stop_candidates if v<r.Close],default=np.nan)

    return {
        "趨勢":tr,"分數":score,"雷達":"｜".join(radar) if radar else "—",
        "狀態":action,"收盤":r.Close,"5MA":r.MA5,"10MA":r.MA10,"20MA":r.MA20,"60MA":r.MA60,
        "量比":r.VOLR,"20MA乖離":gap20,"參考防守":stop,"標籤":"、".join(tags),
        "進貨量次數":int(accumulation),"追高警示":chase,"大量長黑":danger
    }

with st.sidebar:
    token=st.text_input("FinMind Token（可留空）",type="password")
    days=st.slider("資料天數",180,500,300,20)
    st.markdown("**1.2 原則**")
    st.caption("只做技術面。先判斷趨勢，再看均線、量價與K線觸發；分數只排序，不代表買進。")
    min_score=st.slider("雷達最低排序分",0,100,50,5)
    only_bull=st.checkbox("候選只看多頭結構",value=True)


POOL_LARGE="""2330
2317
2454
2303
2382
3231
2379
3034
3711
2412
2881
2882
2886
2891
2892
1301
1303
2002
1216
5871"""

POOL_ACTIVE="""2409
3481
2344
6770
2603
2609
2615
2618
2610
2356
2357
2324
2376
2474
2353
2354
1590
1476
2105
9910"""



POOL_TECH="""2330
2317
2454
2303
2382
3231
2379
3034
3711
2409
3481
2344
6770
2603
2609
2615
2618
2356
2357
2324
2376
2474
2353
2354
1590
1476
2105
9910
2002
1301
1303
1216
1101
1102
2207
2201
2881
2882
2884
2886
2891
2892
5871
6505
1326
3008
3443
2610
2801
2912"""

source=st.radio("掃描模式",["📚 內建選股池","✍️ 手動檢測"],horizontal=True)

if source=="📚 內建選股池":
    pool_name=st.selectbox("選擇股票池",["大型/權值股","活潑/趨勢股","技術雷達50","綜合池"])
    if pool_name=="大型/權值股":
        stocks=[s for s in POOL_LARGE.splitlines() if s.strip()]
    elif pool_name=="活潑/趨勢股":
        stocks=[s for s in POOL_ACTIVE.splitlines() if s.strip()]
    elif pool_name=="技術雷達50":
        stocks=[s for s in POOL_TECH.splitlines() if s.strip()]
    else:
        stocks=list(dict.fromkeys([s for s in (POOL_TECH+"\n"+POOL_LARGE+"\n"+POOL_ACTIVE).splitlines() if s.strip()]))
    st.caption(f"目前選股池：{pool_name}｜{len(stocks)} 檔")
    with st.expander("查看本次股票池"):
        st.code("\n".join(stocks))
else:
    manual_default="2330\n2317"
    stocks=[s.strip() for s in st.text_area(
        "手動檢測｜一行一個股票代號",
        manual_default,height=180,
        help="可以只輸入一檔，也可以一次貼多檔。"
    ).splitlines() if s.strip()]
    st.caption(f"手動檢測：{len(stocks)} 檔")

if st.button("📡 啟動趨勢雷達",type="primary",use_container_width=True):
    if not stocks:
        st.warning("請至少輸入一個股票代號。"); st.stop()
    end=date.today(); start=end-timedelta(days=days+100)
    rows=[]; detail={}; errors=[]
    bar=st.progress(0)
    for i,sid in enumerate(stocks):
        try:
            d=get_price(sid,start,end,token)
            if len(d)<100:
                errors.append(f"{sid}: 歷史資料不足"); continue
            x=indicators(d).dropna().reset_index(drop=True)
            if len(x)<30:continue
            a=analyze(x); a["代號"]=sid
            rows.append(a); detail[sid]=x
        except Exception as e:
            errors.append(f"{sid}: {type(e).__name__}: {e}")
        bar.progress((i+1)/len(stocks),text=f"掃描 {i+1}/{len(stocks)}")
    bar.empty()

    if not rows:
        st.error("沒有成功完成分析。")
        if errors:
            with st.expander("錯誤診斷"):st.code("\n".join(errors[:20]))
        st.stop()

    out=pd.DataFrame(rows).sort_values(["分數","量比"],ascending=False)
    cols=["代號","趨勢","分數","雷達","狀態","收盤","量比","20MA乖離","參考防守","標籤"]
    show=out[cols].copy()
    show["20MA乖離"]=show["20MA乖離"].map(lambda v:f"{v:.1%}")
    show["量比"]=show["量比"].map(lambda v:f"{v:.2f}")
    st.subheader("🏆 今日型態雷達")
    st.dataframe(show,use_container_width=True,hide_index=True)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("多頭",int((out.趨勢=="多頭").sum()))
    c2.metric("突破",int(out.雷達.str.contains("突破",regex=False).sum()))
    c3.metric("回檔轉強",int(out.雷達.str.contains("回檔",regex=False).sum()))
    c4.metric("追高警示",int(out.追高警示.sum()))

    st.subheader("🎯 候選詳解")
    candidates=out[out["狀態"].str.contains("候選|觀察",regex=True) & (out["分數"]>=min_score)]
    if only_bull:
        candidates=candidates[candidates["趨勢"]=="多頭"]
    candidates=candidates.head(12)
    if candidates.empty:st.info("目前沒有符合條件的多方候選。")
    for _,r in candidates.iterrows():
        with st.expander(f'{r.代號}｜{r.趨勢}｜{r.分數}分｜{r.狀態}'):
            a,b,c,d=st.columns(4)
            a.metric("收盤",f'{r.收盤:.2f}')
            b.metric("量比",f'{r.量比:.2f}')
            c.metric("20MA乖離",f'{r["20MA乖離"]:.1%}')
            d.metric("參考防守", "-" if pd.isna(r.參考防守) else f'{r.參考防守:.2f}')
            st.write("**型態：**",r.雷達)
            st.write("**成立條件：**",r.標籤 if r.標籤 else "尚無完整觸發")
            st.caption("參考防守是技術結構提示，不是個人化停損建議。")

    if errors:
        with st.expander(f"資料診斷（{len(errors)}）"):st.code("\n".join(errors[:30]))

    st.download_button("下載雷達結果 CSV",out.to_csv(index=False).encode("utf-8-sig"),
                       "jiaming_trend_radar.csv","text/csv")

st.divider()
st.caption("家銘趨勢雷達 1.2｜研究用途，不構成投資建議。下一階段將加入自動股票池與各型態獨立歷史驗證。")
