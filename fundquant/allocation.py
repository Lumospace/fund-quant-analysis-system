# -*- coding: utf-8 -*-
"""大类资产配置模块（跨资产：股/债/黄金/现金）。

基于代表性 ETF 真实行情，提供多种配置方法（等权 / 股债6-4 / 风险平价 /
目标波动 / 最大夏普 / 最小方差），并支持周期性再平衡回测，对比"再平衡"与
"买入持有"的样本外表现，帮助判断调节奏是否带来风险收益改善。
"""
import numpy as np
import pandas as pd

import config
from . import data, portfolio as pf


# 配置方法（中文标签 -> 内部标识）
ALLOCATION_METHODS = [
    "等权配置", "股债6/4", "风险平价", "目标波动", "最大夏普", "最小方差",
]

METHOD_KEYS = {
    "等权配置": "equal_weight", "股债6/4": "stock_bond_64",
    "风险平价": "risk_parity", "目标波动": "target_vol",
    "最大夏普": "max_sharpe", "最小方差": "min_variance",
}


def asset_class_returns(start: str = None, end: str = None):
    """拉取大类资产代表性 ETF 行情，返回 (价格字典, 对齐日收益矩阵)。

    价格字典键为资产类别中文名（股票/债券/黄金/现金），收益矩阵列同名。
    任一类别无数据时跳过，最终只保留各类别共同有数据的区间。
    """
    start = start or "20150101"
    end = end or pd.Timestamp.now().strftime("%Y%m%d")
    prices = {}
    for cls_name, code, _cn in config.ASSET_CLASS_ETFS:
        h = data.get_etf_hist(code, start, end)
        if h is None or h.empty or "date" not in h.columns or "close" not in h.columns:
            continue
        s = pd.to_numeric(h["close"], errors="coerce").dropna()
        if len(s) < 20:
            continue
        prices[cls_name] = pd.Series(s.values, index=pd.to_datetime(h["date"]))
    if len(prices) < 2:
        return {}, pd.DataFrame()
    px = pd.DataFrame(prices).dropna()
    ret = px.pct_change().dropna()
    return {k: prices[k] for k in px.columns}, ret


def _is_cash(name: str) -> bool:
    nm = str(name or "")
    return ("现金" in nm) or ("货币" in nm)


def _map_weights(res: dict, names) -> np.ndarray:
    """把 optimize_portfolio 的权重结果对齐到 names 顺序的全长数组。"""
    w = np.zeros(len(names))
    for nm, wv in zip(res.get("names", []), res.get("raw_weights", [])):
        if nm in names:
            w[names.index(nm)] = wv
    return w


def _target_vol_weights(ret: pd.DataFrame, target_vol: float) -> np.ndarray:
    """目标波动率权重：风险资产按波动率倒数配比，组合波动对齐目标值，其余进现金。

    不放杠杆：若风险资产按逆波动配比的组合波动已低于目标，则满仓风险资产。
    """
    names = list(ret.columns)
    n = len(names)
    w = np.zeros(n)
    cash_mask = np.array([_is_cash(nm) for nm in names])
    risky = np.arange(n)[~cash_mask]
    if len(risky) == 0:
        if cash_mask.any():
            w[cash_mask] = 1.0 / float(cash_mask.sum())
        else:
            w[:] = 1.0 / n
        return w
    cov_d = pf._robust_cov(ret).values[np.ix_(risky, risky)]
    vol_d = np.sqrt(np.maximum(np.diag(cov_d), 1e-16))
    inv_vol = 1.0 / vol_d
    w_risk = inv_vol / inv_vol.sum()
    port_vol_annual = float(np.sqrt(w_risk @ cov_d @ w_risk)) * np.sqrt(config.TRADING_DAYS)
    if port_vol_annual <= 1e-8:
        scale = 0.0
    else:
        scale = min(target_vol / port_vol_annual, 1.0)
    w[risky] = w_risk * scale
    cash_remain = 1.0 - float(w.sum())
    if cash_remain > 0 and cash_mask.any():
        w[cash_mask] = cash_remain
    s = w.sum()
    return w / s if s > 0 else w


def solve_weights(ret: pd.DataFrame, method: str, target_vol: float = None) -> np.ndarray:
    """按配置方法求解目标权重（对齐 ret.columns 顺序）。"""
    names = list(ret.columns)
    n = len(names)
    key = METHOD_KEYS.get(method, method)
    if key == "equal_weight":
        return np.full(n, 1.0 / n)
    if key == "stock_bond_64":
        w = np.zeros(n)
        for i, nm in enumerate(names):
            if "股" in nm and "债" not in nm:
                w[i] = 0.6
            elif "债" in nm:
                w[i] = 0.4
        if w.sum() < 1e-9:
            return np.full(n, 1.0 / n)
        return w / w.sum()
    if key == "target_vol":
        return _target_vol_weights(ret, target_vol or 0.08)
    if key == "risk_parity":
        res = pf.optimize_portfolio(ret, mode="risk_parity")
    elif key == "min_variance":
        res = pf.optimize_portfolio(ret, mode="min_variance")
    elif key == "max_sharpe":
        res = pf.optimize_portfolio(ret, mode="max_sharpe")
    else:
        return np.full(n, 1.0 / n)
    return _map_weights(res, names)


def method_perf(ret: pd.DataFrame, method: str, target_vol: float = None):
    """单一方法的目标权重 + 全样本绩效。返回 (权重 dict, stats dict)。"""
    names = list(ret.columns)
    w = solve_weights(ret, method, target_vol)
    weights = {names[i]: round(float(w[i]), 4) for i in range(len(names)) if w[i] > 1e-4}
    return weights, pf.portfolio_stats(w, ret)


def allocation_summary(ret: pd.DataFrame, methods=None, target_vol: float = None) -> pd.DataFrame:
    """各配置方法（全样本权重）绩效对比表，按夏普降序。"""
    methods = methods or ALLOCATION_METHODS
    rows = []
    for m in methods:
        w = solve_weights(ret, m, target_vol)
        stats = pf.portfolio_stats(w, ret)
        rows.append({"方法": m, **stats})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("夏普比率", ascending=False).reset_index(drop=True)
    return df


def rebalance_backtest(ret: pd.DataFrame, method: str, target_vol: float = None,
                       rebalance_days: int = 63, lookback: int = 252) -> dict:
    """周期性再平衡回测（对比买入持有）。

    每 rebalance_days 个交易日，用截至当日的最近 lookback 个交易日收益重估目标
    权重并持有下一段；"买入持有"则用首个窗口的初始权重一直持有不动。两者样本外
    区间（首个训练窗结束后）对齐，返回各自的净值与绩效。
    """
    ret = ret.astype(float).dropna()
    n = ret.shape[1]
    if n < 2 or ret.shape[0] < lookback + rebalance_days:
        return {}

    w_hodl = solve_weights(ret.iloc[:lookback], method, target_vol)
    w_cur = w_hodl.copy()
    port_rb = pd.Series(0.0, index=ret.index, dtype=float)
    port_bh = pd.Series(0.0, index=ret.index, dtype=float)

    for t in range(lookback, ret.shape[0], rebalance_days):
        end = min(t + rebalance_days, ret.shape[0])
        port_rb.iloc[t:end] = ret.iloc[t:end].values @ w_cur
        port_bh.iloc[t:end] = ret.iloc[t:end].values @ w_hodl
        if end < ret.shape[0]:
            w_cur = solve_weights(ret.iloc[end - lookback:end], method, target_vol)

    oos_rb = port_rb.iloc[lookback:]
    oos_bh = port_bh.iloc[lookback:]
    return {
        "再平衡": {"nav": (1 + oos_rb).cumprod(), "stats": pf._returns_stats(oos_rb)},
        "买入持有": {"nav": (1 + oos_bh).cumprod(), "stats": pf._returns_stats(oos_bh)},
    }