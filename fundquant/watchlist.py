# -*- coding: utf-8 -*-
"""自选清单与估值偏离监控模块。

支持添加/移除自选基金（本地 session 持久化）、查看实时行情快照，
以及监控估值偏离：当基金真实 PE 分位与价格分位差异超过阈值时标记偏离。
"""
import datetime
import numpy as np
import pandas as pd

import config
from . import data, indicators as ind


# ==================== 估值偏离计算 ====================

def valuation_deviation(prices: pd.Series, pe_hist: pd.DataFrame,
                        pe_window: int = 252) -> dict:
    pe_pct = ind.valuation_percentile_from_pe(pe_hist)
    px = pd.to_numeric(prices, errors="coerce").dropna()
    if len(px) < pe_window:
        return {"signal": "数据不足", "说明": "价格或PE历史不足"}
    price_pct = float((px <= px.iloc[-1]).mean())

    if np.isnan(pe_pct):
        return {
            "pe_pct": None, "price_pct": round(price_pct, 3),
            "deviation": None, "signal": "无真实PE",
            "说明": "无真实 PE 数据，无法计算偏离",
        }

    deviation = float(price_pct - pe_pct)
    if abs(deviation) < 0.15:
        signal = "正常"
        note = f"价格分位({price_pct:.0%}) ≈ PE分位({pe_pct:.0%})，估值定价一致"
    elif deviation > 0.15:
        signal = "偏高估"
        note = (f"价格分位({price_pct:.0%}) 显著高于 PE分位({pe_pct:.0%})，"
                f"偏离 {deviation:.0%}，市场情绪可能过度乐观")
    else:
        signal = "偏低估"
        note = (f"价格分位({price_pct:.0%}) 显著低于 PE分位({pe_pct:.0%})，"
                f"偏离 {-deviation:.0%}，基本面支撑高于市场定价")
    return {
        "pe_pct": round(pe_pct, 3), "price_pct": round(price_pct, 3),
        "deviation": round(deviation, 3), "signal": signal, "说明": note,
    }


def batch_valuation_deviation(codes_names, fkey) -> pd.DataFrame:
    rows = []
    for code, name in codes_names:
        hist = data.get_fund_hist(code, fkey,
                                  start="20200101",
                                  end=datetime.date.today().strftime("%Y%m%d"))
        if hist is None or hist.empty:
            rows.append({"代码": code, "名称": name, "信号": "无数据", "说明": "行情不可用"})
            continue
        prices = ind.to_price_series(hist)
        pe_hist = data.get_fund_pe_history(code, name) if fkey == "etf" else pd.DataFrame()
        dv = valuation_deviation(prices, pe_hist)
        rows.append({
            "代码": code, "名称": name,
            "价格分位": f"{dv['price_pct']:.0%}" if dv['price_pct'] is not None else "--",
            "PE分位": f"{dv['pe_pct']:.0%}" if dv.get('pe_pct') is not None else "--",
            "偏离": f"{dv['deviation']:+.0%}" if dv.get('deviation') is not None else "--",
            "信号": dv["signal"], "说明": dv["说明"],
        })
    return pd.DataFrame(rows)


# ==================== 自选清单 页面组件 ====================

def render_watchlist_page():
    import streamlit as st

    st.header("自选清单与估值偏离监控")

    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []

    default_items = [
        ("510300", "沪深300ETF"), ("510050", "上证50ETF"),
        ("159915", "创业板ETF"), ("518880", "黄金ETF"),
    ]
    if not st.session_state.watchlist:
        st.session_state.watchlist = list(default_items)

    code, name, fkey = _cur_fund()

    c1, c2 = st.columns([3, 1])
    with c1:
        new_code = st.text_input("基金代码（如 510300）", "", key="wl_code",
                                 placeholder="输入 ETF 或基金代码")
    with c2:
        st.write("")
        st.write("")
        if st.button("添加至自选", key="wl_add"):
            if new_code and not any(c == new_code for c, _ in st.session_state.watchlist):
                new_name = new_code
                try:
                    u = data.get_fund_universe("etf")
                    hit = u[u["code"].astype(str) == new_code]
                    if not hit.empty:
                        new_name = str(hit["name"].iloc[0])
                except Exception:
                    pass
                st.session_state.watchlist.append((new_code, new_name))
                st.rerun()

    st.markdown(f"#### 自选基金（共 {len(st.session_state.watchlist)} 只）")
    if not st.session_state.watchlist:
        st.info("自选清单为空，请在上方输入基金代码添加。")
        return

    for i, (wc, wn) in enumerate(st.session_state.watchlist):
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        with col1:
            st.markdown(f"**{wn}**（{wc}）")
        with col2:
            try:
                u = data.get_fund_universe("etf")
                hit = u[u["code"].astype(str) == wc]
                if not hit.empty:
                    px = hit["price"].iloc[0] if "price" in hit.columns else "--"
                    chg = hit["pct_chg"].iloc[0] if "pct_chg" in hit.columns else "--"
                    chg_str = f"{chg:+.2f}%" if isinstance(chg, (int, float)) else str(chg)
                    st.metric("最新价", f"{px:.4f}" if isinstance(px, (int, float)) else px, chg_str)
                else:
                    st.caption("行情加载中...")
            except Exception:
                st.caption("暂不可用")
        with col3:
            try:
                hist = data.get_fund_hist(wc, "etf", "20200101",
                                          datetime.date.today().strftime("%Y%m%d"))
                if hist is not None and not hist.empty:
                    prices = ind.to_price_series(hist)
                    p_ret = ind.period_return(prices, 21) * 100
                    st.metric("近1月", f"{p_ret:+.2f}%")
                else:
                    st.caption("--")
            except Exception:
                st.caption("--")
        with col4:
            if st.button("移除", key=f"wl_rm_{i}"):
                st.session_state.watchlist.pop(i)
                st.rerun()

    st.markdown("---")

    st.markdown("#### 估值偏离监控")
    st.caption("比较价格分位（价格在历史中的位置）与真实 PE 分位（盈利在历史中的位置），"
               "差异超过 15% 时标记为偏离。")
    with st.spinner("批量扫描自选基金估值偏离..."):
        df = batch_valuation_deviation(st.session_state.watchlist, fkey)
    if df is not None and not df.empty:
        st.dataframe(df, width="stretch", hide_index=True, use_container_width=True,
                     column_config={
                         "信号": st.column_config.TextColumn("信号",
                             help="偏高估/偏低估/正常/无真实PE/数据不足"),
                         "说明": st.column_config.TextColumn("说明", width="large"),
                     })
        deviated = df[df["信号"].isin(["偏高估", "偏低估"])]
        if not deviated.empty:
            st.warning(f"发现 {len(deviated)} 只基金存在估值偏离，请关注明细。")
        else:
            st.success("当前自选基金估值均处于正常范围。")
    else:
        st.info("暂无估值数据。")
    st.caption(config.RISK_DISCLAIMER)


def _cur_fund():
    import streamlit as st
    try:
        code = st.session_state.get("code", "510300")
        name = st.session_state.get("name", "沪深300ETF")
        fc = st.session_state.get("fund_class", "场内ETF")
        fkey = config.FUND_CLASS_KEY.get(fc, "etf")
        return code, name, fkey
    except Exception:
        return "510300", "沪深300ETF", "etf"