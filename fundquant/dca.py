# -*- coding: utf-8 -*-
"""定投专项测算子模块。

支持：定期定额 / 估值不定额（越跌越买）两种模式回测；
输出总投入、期末资产、IRR、最大回撤、胜率、微笑曲线；
提供时点优选建议与三种止盈策略（目标收益/估值高位/回撤）模拟对比。
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq

import config
from . import indicators as ind


FREQ_MAP = {"周": "W", "双周": "2W", "月": "ME"}


def _xirr(amounts: list, dates: list) -> float:
    """通过现金流与日期计算年化内部收益率（XIRR）。"""
    amounts = np.asarray(amounts, dtype=float)
    dates = pd.to_datetime(dates)
    if len(amounts) < 2 or amounts[-1] <= 0:
        return np.nan
    d0 = dates[0]
    days = np.array([(d - d0).days for d in dates], dtype=float)

    def npv(rate):
        return float(np.sum(amounts / (1 + rate) ** (days / 365.0)))

    lo, hi = -0.999, 100.0
    try:
        if npv(lo) * npv(hi) < 0:
            return float(brentq(npv, lo, hi, xtol=1e-8))
        return np.nan
    except Exception:
        return np.nan


def _valuation_pct(prices: pd.Series, pe_hist: pd.DataFrame = None) -> pd.Series:
    """估值分位序列（对齐价格索引，0-1）：基于真实 PE-TTM 分位（扩张窗口、无前视）。

    无真实 PE 数据时返回空序列（不再用价格分位冒充估值）。
    """
    if pe_hist is None or getattr(pe_hist, "empty", True):
        return pd.Series(dtype=float)
    pct = ind.pe_percentile_series(pe_hist)
    if pct is None or pct.empty:
        return pd.Series(dtype=float)
    # 前向填充对齐到价格交易日
    idx = pd.to_datetime(pd.Series(prices.index))
    left = pd.DataFrame({"date": idx}).sort_values("date").reset_index(drop=True)
    right = pct.rename("vp").reset_index().rename(columns={"index": "date"}).sort_values("date")
    merged = pd.merge_asof(left, right, on="date", direction="backward")
    return pd.Series(merged["vp"].values, index=pd.to_datetime(merged["date"]))


def dca_backtest(prices: pd.Series, start: str = None, end: str = None,
                 freq: str = "月", amount: float = 1000.0,
                 mode: str = "固定", initial_amount: float = 0.0,
                 pe_hist: pd.DataFrame = None) -> dict:
    """定投回测。mode: 固定(定期定额) / 估值不定额。initial_amount: 起投首笔一次性投入。

    pe_hist: 真实 PE-TTM 历史（date, pe），用于"估值不定额"模式；缺失时该模式退化为固定定额。
    """
    s = pd.to_numeric(prices, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    if start:
        ts = pd.to_datetime(start)
        s = s[s.index >= ts]
    if end:
        s = s[s.index <= pd.to_datetime(end)]
    if len(s) < 20:
        return {}

    # 定投触发日：按频率取每月/每周首个交易日
    freq_key = FREQ_MAP.get(freq, "ME")
    groups = s.groupby(pd.Grouper(freq=freq_key))
    invest_dates = []
    for _, g in groups:
        if len(g) > 0:
            invest_dates.append(ldate := g.index[0])
    invest_dates = [d for d in invest_dates if d in s.index]
    if not invest_dates:
        return {}

    val_pct = _valuation_pct(s, pe_hist)
    shares = 0.0
    cost = 0.0
    cashflows = []
    cf_dates = []
    smile_rows = []

    # 首笔一次性投入
    if initial_amount and initial_amount > 0:
        d0 = s.index[0]
        p0 = float(s.iloc[0])
        shares += initial_amount / p0
        cost += initial_amount
        cashflows.append(-initial_amount)
        cf_dates.append(d0)
        smile_rows.append({"date": d0, "累计投入": cost, "当前市值": shares * p0,
                           "盈亏": shares * p0 - cost})

    for d in invest_dates:
        price = float(s.loc[d])
        amt = amount
        if mode == "估值不定额":
            if val_pct.empty:
                amt = amount  # 无真实 PE 数据，退化为固定定额（与文档一致，而非误投 0.8 倍）
            else:
                vp = val_pct.get(d, np.nan)
                if pd.isna(vp):
                    amt = amount  # 该期估值缺失，按固定金额
                elif vp < 0.2:
                    amt = amount * 1.8   # 估值越低加仓越多
                elif vp < 0.5:
                    amt = amount * 1.2
                elif vp < 0.8:
                    amt = amount * 0.8
                else:
                    amt = 0.0           # 估值高位暂停
        if amt <= 0:
            continue
        shares += amt / price
        cost += amt
        cashflows.append(-amt)
        cf_dates.append(d)
        value = shares * price
        smile_rows.append({"date": d, "累计投入": cost, "当前市值": value,
                           "盈亏": value - cost})

    if shares == 0:
        return {}

    final_price = float(s.iloc[-1])
    final_value = shares * final_price
    cashflows.append(final_value)
    cf_dates.append(s.index[-1])
    irr = _xirr(cashflows, cf_dates)

    # 组合净值（定投）用于回撤/胜率
    smile = pd.DataFrame(smile_rows).set_index("date")
    nav = smile["当前市值"] / smile["累计投入"].replace(0, np.nan)
    nav = nav.dropna()
    mdd = float((nav / nav.cummax() - 1).min()) if len(nav) > 1 else 0.0

    return {
        "mode": mode, "freq": freq, "每期金额": amount,
        "定投期数": len(cashflows) - 1,
        "总投入": round(cost, 2),
        "期末资产": round(final_value, 2),
        "累计收益": round(final_value - cost, 2),
        "收益率": round((final_value / cost - 1) * 100, 2),
        "定投IRR": round(irr * 100, 2) if not np.isnan(irr) else None,
        "最大回撤": round(mdd * 100, 2),
        "微笑曲线": smile,
    }


def dca_stop_profit_backtest(prices: pd.Series, start: str = None, end: str = None,
                             freq: str = "月", amount: float = 1000.0,
                             pe_hist: pd.DataFrame = None) -> dict:
    """三种止盈策略模拟对比（目标收益 / 估值高位 / 回撤止盈）。

    pe_hist: 真实 PE-TTM 历史，用于"估值高位"止盈；缺失时该策略退化为中性触发。
    """
    s = pd.to_numeric(prices, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    if start:
        s = s[s.index >= pd.Timestamp(start)]
    if end:
        s = s[s.index <= pd.Timestamp(end)]
    if len(s) < 20:
        return {}

    groups = s.groupby(pd.Grouper(freq=FREQ_MAP.get(freq, "ME")))
    invest_dates = [g.index[0] for _, g in groups if len(g) > 0]
    invest_dates = [d for d in invest_dates if d in s.index]
    val_pct = _valuation_pct(s, pe_hist)

    def simulate(strategy: str, target: float):
        shares = 0.0
        cost = 0.0          # 累计已投入本金
        realized = 0.0      # 止盈清仓变现所得
        peak_value = 0.0    # 持仓市值历史峰值（用于回撤止盈）
        sold = False
        for d in invest_dates:
            if sold:
                break  # 一揽子止盈：清仓后不再定投
            price = float(s.loc[d])
            shares += amount / price
            cost += amount
            mv = shares * price
            peak_value = max(peak_value, mv)
            vp = val_pct.get(d, np.nan) if not val_pct.empty else np.nan
            vp = 0.5 if pd.isna(vp) else vp
            trigger = False
            if strategy == "目标收益" and mv >= cost * (1 + target):
                trigger = True
            elif strategy == "估值高位" and vp > target:
                trigger = True
            elif strategy == "回撤止盈" and mv < peak_value * (1 - target):
                trigger = True
            if trigger:
                realized = mv
                sold = True
        # 期末结算：已止盈 = 清仓现金；未止盈 = 期末持仓市值
        final_mv = realized if sold else shares * float(s.iloc[-1])
        total_cost = cost  # 真实累计投入（止盈策略为清仓前投入）
        return {
            "策略": strategy, "参数": f"{target*100:.0f}%",
            "期末资产": round(final_mv, 2),
            "总投入": round(total_cost, 2),
            "模拟收益率": round((final_mv / total_cost - 1) * 100, 2) if total_cost else 0.0,
        }

    rows = [
        simulate("目标收益", 0.30),
        simulate("估值高位", 0.80),
        simulate("回撤止盈", 0.10),
    ]
    return {"止盈对比": pd.DataFrame(rows)}


def dca_timing_advice(prices: pd.Series, pe_hist: pd.DataFrame = None) -> dict:
    """定投时点优选建议：基于真实估值分位 + 波动因子。无真实 PE 数据时提示数据不足。"""
    s = pd.to_numeric(prices, errors="coerce").dropna()
    vp_series = _valuation_pct(s, pe_hist)
    vol = ind.annualized_volatility(s)
    if vp_series.empty or pd.isna(vp_series.iloc[-1]):
        return {"估值分位": None, "波动率": round(vol * 100, 1),
                "建议": "无真实 PE-TTM 估值数据（该基金不跟踪可解析的宽基指数），建议按普通定额定投"}
    vp = vp_series.iloc[-1]
    if vp < 0.2:
        advice = "当前处于历史估值低位，建议加码定投（每次 1.5-2 倍）"
    elif vp < 0.5:
        advice = "估值中性偏低，建议维持正常定投"
    elif vp < 0.8:
        advice = "估值中性偏高，建议减量定投（每次 0.5-0.8 倍）"
    else:
        advice = "估值高位，建议暂停定投或止盈减仓，等待回调"
    return {"估值分位": round(vp * 100, 1), "波动率": round(vol * 100, 1), "建议": advice}