# -*- coding: utf-8 -*-
"""历史回测与策略对比子模块。

在同一只基金的历史行情上并行运行多套经典策略，输出统一的归一化净值、
绩效指标（总收益/年化收益/年化波动/夏普/最大回撤/Calmar/交易次数），
支持横向对比与净值/回撤叠加可视化。

内置五套策略：
  1. 买入持有（基准）          —— 全程满仓，无择时；
  2. 双均线趋势                —— 短均线上穿长均线满仓，下穿空仓；
  3. 动量择时                  —— 过去 N 日收益为正持有，否则空仓；
  4. 定期定额定投              —— 复用 dca 模块（固定金额）；
  5. 估值定投                  —— 复用 dca 模块（估值不定额，越跌越买）。

择时类策略统一计入交易成本（单边手续费 + 滑点，费率由调用方传入），
信号严格基于截至前一交易日的信息生成，保证无前视偏差。
"""
import numpy as np
import pandas as pd

import config
from . import dca


def _nav_stats(nav: pd.Series, trades: int = 0, ann: float = None) -> dict:
    """由归一化净值序列计算统一绩效指标（收益类均为小数，如 0.15 表示 15%）。

    ann: 外部给定的年化收益（如定投 IRR），给定则覆盖净值年化口径。
    """
    nav = pd.to_numeric(nav, errors="coerce").dropna()
    if len(nav) < 2 or nav.iloc[0] <= 0:
        return {}
    # 年化年限统一按交易日数折算，与年化波动率 sqrt(252) 口径一致
    yrs = (len(nav) - 1) / config.TRADING_DAYS

    total = float(nav.iloc[-1] / nav.iloc[0] - 1)
    if ann is None:
        ann = float(nav.iloc[-1] / nav.iloc[0]) ** (1.0 / yrs) - 1 if yrs > 0 else np.nan
    r = nav.pct_change().dropna()
    vol = float(r.std(ddof=1) * np.sqrt(config.TRADING_DAYS)) if len(r) > 1 else 0.0
    sharpe = float((ann - config.RISK_FREE_RATE) / vol) if vol and vol > 0 else np.nan
    mdd = float((nav / nav.cummax() - 1).min())
    calmar = float(ann / abs(mdd)) if mdd is not None and mdd < 0 else np.nan
    return {
        "总收益": total,
        "年化收益": float(ann) if ann is not None else np.nan,
        "年化波动": vol,
        "夏普比率": sharpe,
        "最大回撤": mdd,
        "Calmar": calmar,
        "交易次数": int(trades),
    }


def run_signal_backtest(prices: pd.Series, signal: pd.Series,
                        cost_rate: float = 0.00025, name: str = "") -> dict:
    """信号类策略回测引擎。

    signal[t] 表示 t 日应持有的目标仓位（0~1），必须基于 <=t-1 日信息计算。
    组合日收益 = 当日仓位×当日涨跌 − 换仓幅度×单边交易成本。
    """
    s = pd.to_numeric(prices, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    if len(s) < 2:
        return {"name": name, "nav": pd.Series(dtype=float), "stats": {}}
    sig = signal.reindex(s.index).fillna(0.0).astype(float).clip(0.0, 1.0)

    r = s.pct_change().fillna(0.0)  # 首日无涨跌，且须保留首期建仓成本
    dw = sig.diff()                 # 每期换仓幅度
    dw.iloc[0] = sig.iloc[0]        # 首期建仓成本
    port_ret = sig * r - dw.abs() * cost_rate
    port_ret = port_ret.fillna(0.0)
    nav = (1.0 + port_ret).cumprod()
    trades = int((dw.abs() > 1e-6).sum())
    return {"name": name, "nav": nav, "stats": _nav_stats(nav, trades=trades)}


def _dca_result(res: dict, prices_s: pd.Series, name: str) -> dict:
    """把 dca 模块回测结果转成统一回测结果（nav 归一化 + 统一指标）。"""
    if not res or "微笑曲线" not in res:
        return {"name": name, "nav": pd.Series(dtype=float), "stats": {}}
    smile = res["微笑曲线"]
    cost = float(res.get("总投入", 0.0))
    final_value = float(res.get("期末资产", 0.0))
    if cost <= 0 or smile.empty:
        return {"name": name, "nav": pd.Series(dtype=float), "stats": {}}

    nav = smile["当前市值"] / smile["累计投入"].replace(0, np.nan)
    nav = pd.to_numeric(nav, errors="coerce").dropna()
    # 追加期末结算点（定投点之间无记录，期末补一点保证口径同信号类一致）
    try:
        nav.loc[pd.to_datetime(prices_s.index[-1])] = final_value / cost
    except Exception:
        pass

    irr = res.get("定投IRR")
    ann_ov = irr / 100.0 if isinstance(irr, (int, float)) else None
    stats = _nav_stats(nav, trades=int(res.get("定投期数", 0)), ann=ann_ov)
    stats["总收益"] = final_value / cost - 1  # 定投总收益以真实累计投入为准
    return {"name": name, "nav": nav, "stats": stats}


def compare_strategies(prices: pd.Series, pe_hist: pd.DataFrame = None,
                       ma_fast: int = 20, ma_slow: int = 60, mom_window: int = 60,
                       cost_rate: float = 0.00025, dca_freq: str = "月",
                       dca_amount: float = 1000.0,
                       start: str = None, end: str = None) -> tuple:
    """对单只基金并行回测五套策略，返回 (有效结果dict, 汇总DataFrame)。"""
    s = pd.to_numeric(prices, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    if start:
        s = s[s.index >= pd.Timestamp(start)]
    if end:
        s = s[s.index <= pd.Timestamp(end)]
    if len(s) < 20:
        return {}, pd.DataFrame()

    results = {}
    # 1. 买入持有
    results["买入持有"] = run_signal_backtest(s, pd.Series(1.0, index=s.index),
                                              cost_rate, "买入持有")
    # 2. 双均线趋势（均线使用截至前一交易日数据）
    fast = s.rolling(int(ma_fast)).mean()
    slow = s.rolling(int(ma_slow)).mean()
    results[f"双均线趋势({ma_fast}/{ma_slow})"] = run_signal_backtest(
        s, (fast.shift(1) > slow.shift(1)).astype(float), cost_rate,
        f"双均线趋势({ma_fast}/{ma_slow})")
    # 3. 动量择时
    mom = s.pct_change(int(mom_window)).shift(1)
    results[f"动量择时({mom_window}日)"] = run_signal_backtest(
        s, (mom > 0).astype(float), cost_rate, f"动量择时({mom_window}日)")
    # 4/5. 定投类（复用 dca 模块）
    results["定期定额定投"] = _dca_result(
        dca.dca_backtest(s, None, None, dca_freq, dca_amount, "固定", 0.0, pe_hist),
        s, "定期定额定投")
    results["估值定投"] = _dca_result(
        dca.dca_backtest(s, None, None, dca_freq, dca_amount, "估值不定额", 0.0, pe_hist),
        s, "估值定投")

    # 统一结果内 name 与字典键一致，避免绘图图例为空/覆盖
    for k, v in results.items():
        v["name"] = k

    valid = {k: v for k, v in results.items() if v.get("stats")}
    return valid, _summary(valid)


def _summary(results: dict) -> pd.DataFrame:
    """结果汇总表（数值口径：收益/波动/回撤为小数，供 UI 格式化）。"""
    rows = []
    for name, r in results.items():
        st = r["stats"]
        rows.append({
            "策略": name,
            "总收益": st.get("总收益", np.nan),
            "年化收益": st.get("年化收益", np.nan),
            "年化波动": st.get("年化波动", np.nan),
            "夏普比率": st.get("夏普比率", np.nan),
            "最大回撤": st.get("最大回撤", np.nan),
            "Calmar": st.get("Calmar", np.nan),
            "交易次数": st.get("交易次数", np.nan),
        })
    return pd.DataFrame(rows)