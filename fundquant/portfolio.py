# -*- coding: utf-8 -*-
"""投资组合量化优化模块（核心落地功能）。

马科维茨均值-方差模型：最小方差 / 最大夏普 / 目标收益 / 有效前沿，
支持权重边界约束（单资产/单行业上限）。附带组合 VaR / CVaR 测算。
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config
from . import indicators as ind


def _robust_cov(ret: pd.DataFrame) -> pd.DataFrame:
    """稳健协方差估计：Ledoit-Wolf 收缩，降低样本协方差的病态条件数。

    直接使用样本协方差在资产较多/样本不足时会放大估计误差，使优化器
    （SLSQP）因数值不稳定产出集中在少数资产的极端权重。Ledoit-Wolf 是
    业界标准的收缩正则化方法，且保证半正定；失败时退化为样本协方差。
    """
    r = ret.dropna(how="any")
    if r.shape[0] < 2 or r.shape[1] < 1:
        return ret.cov()
    try:
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(r).covariance_
        return pd.DataFrame(cov, index=r.columns, columns=r.columns)
    except Exception:
        return r.cov()


def build_return_matrix(prices_dict: dict) -> tuple:
    """输入 {名称: 价格Series}，输出对齐后的收益矩阵 + 协方差 + 期望收益。"""
    closes = {}
    for name, s in prices_dict.items():
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) > 5:
            closes[name] = s
    if not closes:
        return None, None, None
    px = pd.DataFrame(closes)
    # 先对齐到所有资产共有的交易日起止区间；区间内少量停牌缺失日做前向填充，
    # 避免「单个资产某日缺失即丢弃当日全部资产」导致截面无谓损失。
    common = px.dropna(how="any")
    if common.empty:
        return None, None, None
    px = px.loc[common.index.min():common.index.max()].ffill().dropna(how="any")
    ret = px.pct_change().dropna()
    return ret, _robust_cov(ret) * config.TRADING_DAYS, ret.mean() * config.TRADING_DAYS


def _portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(w.T @ cov @ w))


def _portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    return float(w @ mu)


def _neg_sharpe(w, mu, cov, rf):
    w = np.asarray(w)
    ret = _portfolio_return(w, mu)
    vol = _portfolio_vol(w, cov)
    if vol == 0:
        return 1e9
    return -((ret - rf) / vol)


def _optimize(mu, cov, bounds, rf, objective, target=None, eq_cons=None, ineq_cons=None):
    n = len(mu)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target is not None:
        cons.append({"type": "eq", "fun": lambda w: _portfolio_return(w, mu) - target})
    if eq_cons:
        cons += eq_cons
    if ineq_cons:
        cons += ineq_cons

    lower = float(bounds[0][0]) if bounds else 0.0
    upper = float(bounds[0][1]) if bounds else 1.0
    rng = np.random.default_rng(0)
    starts = [np.ones(n) / n]
    for _ in range(24):
        starts.append(rng.dirichlet(np.ones(n)))

    best_w, best_obj = None, np.inf
    fallback_w, fallback_err = None, np.inf
    for raw in starts:
        w0 = np.clip(np.asarray(raw, dtype=float), lower, upper)
        s = w0.sum()
        w0 = w0 / s if s > 0 else np.ones(n) / n
        try:
            res = minimize(objective, w0, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
        except Exception:
            continue
        x = np.asarray(res.x, dtype=float)
        if res.success:
            o = float(objective(x))
            if o < best_obj:
                best_obj, best_w = o, x
        else:
            # 失败时记录约束违反最小的点，作为次优兜底（而非平均权重）
            err = abs(np.sum(x) - 1.0)
            if target is not None:
                err += abs(_portfolio_return(x, mu) - target)
            if err < fallback_err:
                fallback_err, fallback_w = err, x

    if best_w is not None:
        return best_w
    if fallback_w is not None:
        w = np.clip(np.asarray(fallback_w, dtype=float), lower, upper)
        return w / w.sum() if w.sum() > 0 else np.ones(n) / n
    return np.ones(n) / n


def _build_sector_constraints(names, sector_limit):
    """生成行业/板块集中度上限约束（scipy 不等式约束）。

    「其他」板块（未命中关键词）视为无明确行业，不纳入集中度约束。
    返回 (约束列表, 板块分组 dict, 可行性提示 note)。
    """
    from . import data as _data
    groups = {}
    for i, nm in enumerate(names):
        sec = _data.classify_sector(nm)
        if sec == "其他":
            continue
        groups.setdefault(sec, []).append(i)

    eff = sector_limit
    note = None
    k = len(groups)
    if k > 0 and sector_limit * k < 1 - 1e-9:
        eff = 1.0 / k
        note = (f"行业集中度上限 {sector_limit*100:.0f}% 过低（{k} 个板块最多合计 "
                f"{sector_limit*k*100:.0f}%，不足 100%），已自动放宽至 {eff*100:.0f}%")

    cons = []
    for sec, idxs in groups.items():
        idxs = np.asarray(idxs, dtype=int)
        cons.append({"type": "ineq",
                     "fun": lambda w, idxs=idxs, lim=eff: lim - float(np.sum(np.asarray(w)[idxs]))})
    return cons, groups, note


def _risk_parity_weights(cov: np.ndarray, bounds, ineq_cons=None, budget=None) -> np.ndarray:
    """风险平价 / 风险预算权重。

    风险平价（budget=None）：每个资产风险贡献相等（1/n），即等风险贡献 ERC。
    风险预算（budget 给定）：每个资产风险贡献 = 预设预算比例（自动归一化）。
    二者都最小化各资产实际风险贡献与目标预算之差的平方和，比马科维茨更稳健、
    不易集中于单一资产。
    """
    n = cov.shape[0]
    b = np.full(n, 1.0 / n)
    if budget is not None:
        arr = np.asarray(budget, dtype=float)
        if arr.shape == (n,) and arr.sum() > 0:
            b = arr / arr.sum()

    def obj(w):
        w = np.asarray(w, dtype=float)
        pvar = float(w @ cov @ w)
        if pvar <= 1e-14:
            return 1e9
        rc = w * (cov @ w) / pvar
        return float(np.sum((rc - b) ** 2))

    rng = np.random.default_rng(0)
    starts = [np.ones(n) / n] + [rng.dirichlet(np.ones(n)) for _ in range(24)]
    lower = float(bounds[0][0]) if bounds else 0.0
    upper = float(bounds[0][1]) if bounds else 1.0
    best_w, best_obj = None, np.inf
    for raw in starts:
        w0 = np.clip(np.asarray(raw, dtype=float), lower, upper)
        s = w0.sum()
        w0 = w0 / s if s > 0 else np.ones(n) / n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        if ineq_cons:
            cons += ineq_cons
        try:
            res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                           options={"maxiter": 1000, "ftol": 1e-12})
        except Exception:
            continue
        if res.success:
            o = float(obj(res.x))
            if o < best_obj:
                best_obj, best_w = o, res.x
    if best_w is None:
        best_w = np.ones(n) / n
    return np.asarray(best_w, dtype=float)


def _expected_returns(ret: pd.DataFrame) -> np.ndarray:
    """期望收益估计（年化）：样本均值向截面均值做 James-Stein 收缩。

    样本均值作为期望收益估计极不稳定（估计误差 >> 真实信号），直接喂给
    均值-方差会产出对历史噪声过拟合的极端/突兀权重。将每个资产的样本均值
    按 config.EXPECTED_RETURN_SHRINK 强度向截面均值收缩，量级更稳、
    排序关系不变；风险平价/风险预算/最小方差只依赖协方差，不受本函数影响。
    """
    mu = ret.mean().values * config.TRADING_DAYS
    mu_bar = float(np.mean(mu))
    return mu_bar + (1.0 - config.EXPECTED_RETURN_SHRINK) * (mu - mu_bar)


def optimize_portfolio(ret: pd.DataFrame, mode: str = "min_variance",
                       target_return: float = None, max_weight: float = 1.0,
                       bounds=None, sector_limit=None, budget=None,
                       return_frontier: bool = False):
    """投资组合优化。

    mode: min_variance / max_sharpe / target_return / risk_parity / risk_budgeting
    max_weight: 单资产权重上限（bounds 未提供时使用）
    bounds: 逐资产边界 [(min,max), ...]，长度须等于资产数，优先于 max_weight
    sector_limit: 行业/板块集中度上限（0~1），None 表示不限制
    budget: 风险预算模式下各资产目标风险贡献比例（长度=资产数，自动归一化；缺省等额）
    返回结果字典。
    """
    mu = _expected_returns(ret)
    cov = _robust_cov(ret).values * config.TRADING_DAYS
    names = list(ret.columns)
    n = len(names)
    rf = config.RISK_FREE_RATE
    notes = []

    if bounds is None:
        # 单资产上限过低会导致无可行解（max_weight*n < 1），此时放宽并提示
        upper = max_weight
        if max_weight * n < 1 - 1e-9:
            upper = 1.0 / n
            notes.append(f"单资产上限 {max_weight*100:.0f}% 过低（{n} 只资产合计最多 "
                         f"{max_weight*n*100:.0f}%，不足 100%），已自动放宽至等权上限 {upper*100:.0f}%")
        bounds = [(0.0, upper)] * n
    else:
        bounds = [tuple(b) for b in bounds]
        lo_sum = sum(max(0.0, b[0]) for b in bounds)
        hi_sum = sum(max(0.0, b[1]) for b in bounds)
        if lo_sum > 1 + 1e-9 or hi_sum < 1 - 1e-6:
            upper = 1.0 / n
            if lo_sum > 1 + 1e-9:
                notes.append(f"逐资产下界合计 {lo_sum*100:.0f}% 超过 100%，无可行解，"
                             f"已回退等权上限 {upper*100:.0f}%")
            else:
                notes.append(f"逐资产上界合计 {hi_sum*100:.0f}% 不足 100%，无可行解，"
                             f"已回退等权上限 {upper*100:.0f}%")
            bounds = [(0.0, upper)] * n

    ineq_cons = []
    if sector_limit is not None and sector_limit < 1.0 - 1e-9:
        ineq_cons, _, snote = _build_sector_constraints(names, sector_limit)
        if snote:
            notes.append(snote)

    def obj_minvar(w):
        return _portfolio_vol(w, cov)

    if mode == "risk_parity":
        w = _risk_parity_weights(cov, bounds, ineq_cons)
    elif mode == "risk_budgeting":
        w = _risk_parity_weights(cov, bounds, ineq_cons, budget)
    elif mode == "min_variance":
        w = _optimize(mu, cov, bounds, rf, obj_minvar, ineq_cons=ineq_cons)
    elif mode == "target_return" and target_return is not None:
        lo, hi = float(mu.min()), float(mu.max())
        if target_return > hi:
            notes.append(f"目标年化收益 {target_return*100:.1f}% 超出组合可达上限 {hi*100:.1f}%，已按上限优化")
            target_return = hi
        elif target_return < lo:
            notes.append(f"目标年化收益 {target_return*100:.1f}% 低于组合可达下限 {lo*100:.1f}%，已按下限优化")
            target_return = lo
        w = _optimize(mu, cov, bounds, rf, obj_minvar, target=target_return, ineq_cons=ineq_cons)
    else:  # max sharpe
        w = _optimize(mu, cov, bounds, rf, lambda w: _neg_sharpe(w, mu, cov, rf),
                      ineq_cons=ineq_cons)

    w = _clean_weights(w)

    result = {
        "mode": mode,
        "weights": {names[i]: round(float(w[i]), 4) for i in range(n) if w[i] > 1e-4},
        "raw_weights": w,
        "names": names,
        "stats": portfolio_stats(w, ret),
        "note": "；".join(notes) or None,
    }
    if return_frontier and mode not in ("risk_parity", "risk_budgeting"):
        result["frontier"] = _efficient_frontier(mu, cov, bounds, rf, ineq_cons=ineq_cons)
    return result


def _clean_weights(w):
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0, None)
    s = w.sum()
    if s > 0:
        w = w / s
    return w


def _efficient_frontier(mu, cov, bounds, rf, n_points=40, ineq_cons=None):
    """采样有效前沿（目标收益限制在可达区间内，避免两端点无解）。"""
    lo, hi = float(np.min(mu)), float(np.max(mu))
    pad = max((hi - lo) * 0.02, 1e-4)
    rets = np.linspace(lo + pad, hi - pad, n_points) if hi > lo else np.array([lo])
    pts = []
    for tr in rets:
        w = _optimize(mu, cov, bounds, rf, lambda w: _portfolio_vol(w, cov),
                      target=tr, ineq_cons=ineq_cons)
        w = _clean_weights(w)
        pts.append((float(_portfolio_vol(w, cov)), float(_portfolio_return(w, mu))))
    return pd.DataFrame(pts, columns=["vol", "ret"])


def _returns_stats(pr: pd.Series) -> dict:
    """从组合日收益序列计算统一绩效口径（年化收益/波动/夏普/回撤/VaR/CVaR）。"""
    pr = pd.Series(pr, dtype=float).dropna()
    if len(pr) < 2:
        return {"年化收益": 0.0, "年化波动率": 0.0, "夏普比率": 0.0,
                "最大回撤": 0.0, "VaR95": 0.0, "CVaR95": 0.0}
    nav = (1 + pr).cumprod()
    ann_ret = float((1 + pr).prod()) ** (config.TRADING_DAYS / len(pr)) - 1
    vol = float(pr.std(ddof=1) * np.sqrt(config.TRADING_DAYS))
    sharpe = float((ann_ret - config.RISK_FREE_RATE) / vol) if vol > 0 else 0.0
    mdd = float((nav / nav.cummax() - 1).min())
    var95 = float(-np.percentile(pr, 5))
    tail = pr <= np.percentile(pr, 5)
    cvar95 = float(-pr[tail].mean()) if tail.any() else var95
    return {
        "年化收益": round(ann_ret * 100, 2), "年化波动率": round(vol * 100, 2),
        "夏普比率": round(sharpe, 3), "最大回撤": round(mdd * 100, 2),
        "VaR95": round(var95 * 100, 2), "CVaR95": round(cvar95 * 100, 2),
    }


def portfolio_stats(w: np.ndarray, ret: pd.DataFrame) -> dict:
    """组合绩效（固定权重）：年化收益/波动/夏普/最大回撤/VaR/CVaR。"""
    pr = pd.Series(np.asarray(ret) @ np.asarray(w), index=ret.index)
    return _returns_stats(pr)


def risk_contribution(w: np.ndarray, ret: pd.DataFrame) -> pd.DataFrame:
    """风险贡献分解（波动率归因）。

    按资产把组合年化波动率拆开：边际风险贡献 MRC_i = (Σw)_i / σ_p，
    风险贡献 RC_i = w_i · MRC_i，占比 = RC_i / σ_p（合计 100%）。
    Σ 采用 Ledoit-Wolf 年化协方差，与优化口径一致，避免样本协方差放大归因误差。
    返回 DataFrame：资产 / 权重(%) / 边际风险贡献 / 风险贡献(波动率%) / 风险贡献占比(%)，
    按占比降序排列，便于一眼识别「风险贡献大于权重」的集中资产。
    """
    names = list(ret.columns)
    n = len(names)
    w = np.asarray(w, dtype=float)
    if n == 0 or len(w) != n:
        return pd.DataFrame(columns=["资产", "权重", "边际风险贡献", "风险贡献", "风险贡献占比(%)"])
    cov = _robust_cov(ret).values * config.TRADING_DAYS  # 年化协方差
    covw = cov @ w
    pvar = float(w @ covw)
    pvol = float(np.sqrt(max(pvar, 1e-16)))
    if pvol <= 1e-12:
        mrc = rc = np.zeros(n)
    else:
        mrc = covw / pvol
        rc = w * mrc
    return pd.DataFrame({
        "资产": names,
        "权重": np.round(w * 100, 2),
        "边际风险贡献": np.round(mrc, 4),
        "风险贡献": np.round(rc * 100, 2),
        "风险贡献占比(%)": np.round(rc / pvol * 100, 2) if pvol > 1e-12 else np.zeros(n),
    }).sort_values("风险贡献占比(%)", ascending=False).reset_index(drop=True)


def composite_nav(weights: dict, prices_dict: dict) -> pd.Series:
    """给定权重与价格，返回组合日净值序列（起始归1）。"""
    df = pd.DataFrame()
    for name, w in weights.items():
        if name in prices_dict:
            s = prices_dict[name]
            df[name] = s / s.iloc[0] * w
    if df.empty:
        return pd.Series(dtype=float)
    combo = df.dropna().sum(axis=1)
    return combo / combo.iloc[0]


def drawdown_attribution(weights: dict, prices_dict: dict) -> dict:
    """组合最大回撤归因：定位组合峰→谷区间，分解各资产对该段回撤的贡献。

    组合净值按权重加权合成（口径同 composite_nav）。找出最大回撤的峰值/谷值日期，
    用「峰值时该资产权重 × 资产区间跌幅」近似分解回撤（合计≈组合回撤），
    识别拖累组合回撤的主要资产。返回 {start, end, mdd, contributions}，
    成分不足或回撤区间无效时返回 {}。contributions 为 {资产: 贡献(负值表示拖累，%)}。
    """
    if not weights or not prices_dict:
        return {}
    df = pd.DataFrame()
    for name, w in weights.items():
        if name in prices_dict and w > 0:
            s = pd.to_numeric(prices_dict[name], errors="coerce").dropna()
            if len(s) > 1:
                df[name] = s / s.iloc[0] * w
    if df.shape[1] < 1 or df.shape[0] < 2:
        return {}
    combo = df.dropna().sum(axis=1)
    combo = combo / combo.iloc[0]
    dd = combo / combo.cummax() - 1
    trough = dd.idxmin()
    peak = combo.loc[:trough].idxmax()
    if peak >= trough:
        return {}
    mdd = float(dd.loc[trough])
    combo_peak = float(combo.loc[peak])
    contribs = {}
    for name in df.columns:
        w_peak = float(df[name].loc[peak] / combo_peak) if combo_peak > 0 else 0.0
        a_ret = float(df[name].loc[trough] / df[name].loc[peak] - 1)
        contribs[name] = w_peak * a_ret * 100  # 负值=贡献回撤
    return {
        "start": peak, "end": trough,
        "mdd": round(mdd * 100, 2),
        "contributions": {k: round(v, 2) for k, v in
                          sorted(contribs.items(), key=lambda x: x[1])},
    }


# ============ 尾部风险（蒙特卡洛 + 相关性） ============

def _correlated_tail(ret: pd.DataFrame, weights: dict, alpha: float = 0.95,
                     horizon: int = 1, n_paths: int = 20000, seed: int = 2024):
    """组合相关尾部风险：Ledoit-Wolf 协方差 + Cholesky 多变量蒙特卡洛。

    保留资产间相关结构，模拟 horizon 期联合收益后合成组合收益分布，取尾部
    VaR/CVaR。与「各资产 VaR 简单相加」不同，本方法正确计入相关结构对尾部
    损失的放大/压缩（如高相关时分散化失效、组合尾部更厚）。
    返回 (VaR, CVaR)，均为正数亏损幅度。
    """
    ret = ret.astype(float).dropna()
    names = list(ret.columns)
    n = len(names)
    mu = ret.mean().values
    cov_d = _robust_cov(ret).values          # 日频协方差
    sigma = np.sqrt(np.maximum(np.diag(cov_d), 1e-12))
    w = np.asarray([weights.get(c, 0.0) for c in names], dtype=float)
    if w.sum() > 0:
        w = w / w.sum()
    else:
        w = np.full(n, 1.0 / n)
    try:
        L = np.linalg.cholesky(cov_d)
    except np.linalg.LinAlgError:
        L = np.diag(sigma)                   # 半正定失败时退化为独立（仅对角线）
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, n_paths))
    log_ret = (mu[:, None] - 0.5 * sigma[:, None] ** 2) * horizon \
        + np.sqrt(horizon) * (L @ z)
    sim_ret = np.exp(log_ret) - 1.0
    port_ret = w @ sim_ret
    q = float(np.percentile(port_ret, (1 - alpha) * 100))
    var = float(-q)
    tail = port_ret[port_ret <= q]
    cvar = float(-tail.mean()) if len(tail) else var
    return var, cvar


def portfolio_tail_risk(ret: pd.DataFrame, weights: dict = None, alpha: float = 0.95,
                        horizon: int = 1, n_paths: int = 20000, seed: int = 2024) -> dict:
    """组合尾部风险全景：历史 + 蒙特卡洛（相关）+ 分散化收益。

    返回：
    - 历史 VaR/CVaR：基于组合历史收益分位；
    - 蒙特卡洛 VaR/CVaR：Ledoit-Wolf 协方差 + Cholesky 多变量模拟（保留相关）；
    - 平均相关性、分散化收益（相关组合尾部相对"假设独立"的压缩比例）。
    """
    ret = ret.astype(float).dropna()
    names = list(ret.columns)
    n = len(names)
    if weights is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.asarray([weights.get(c, 0.0) for c in names], dtype=float)
        w = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)

    pr = pd.Series(ret.values @ w, index=ret.index)
    hist_q = float(np.percentile(pr, (1 - alpha) * 100))
    hist_var = float(-hist_q)
    hist_tail = pr[pr <= hist_q]
    hist_cvar = float(-hist_tail.mean()) if len(hist_tail) else hist_var

    mc_var, mc_cvar = _correlated_tail(ret, dict(zip(names, w)), alpha, horizon,
                                       n_paths, seed)

    # 分散化收益：相关组合尾部 vs 假设各资产独立时的加权尾部之和
    ind_sum = 0.0
    for i, c in enumerate(names):
        _, acvar = ind.mc_tail_from_returns(ret[c], alpha, horizon, n_paths, seed + i)
        ind_sum += w[i] * acvar
    div_benefit = float(1.0 - mc_cvar / ind_sum) if ind_sum > 1e-12 else 0.0

    corr = ret.corr().values
    pairs = n * (n - 1) / 2
    avg_corr = float(np.triu(corr, 1).sum() / pairs) if n > 1 and pairs > 0 else 0.0

    return {
        "历史VaR95": round(hist_var * 100, 2),
        "历史CVaR95": round(hist_cvar * 100, 2),
        "蒙特卡洛VaR95": round(mc_var * 100, 2),
        "蒙特卡洛CVaR95": round(mc_cvar * 100, 2),
        "平均相关性": round(avg_corr, 3),
        "分散化收益": round(div_benefit * 100, 2),
        "说明": "蒙特卡洛 VaR/CVaR 基于 Ledoit-Wolf 协方差 + Cholesky 多变量模拟，保留资产相关性"
    }


# ============ 滚动样本外回测（方法横向对比） ============

METHOD_LABELS = {
    "min_variance": "最小方差", "max_sharpe": "最大夏普",
    "risk_budgeting": "风险预算", "equal_weight": "等权基准",
    "target_return": "目标收益",
}


def _solve_weights(ret: pd.DataFrame, mode: str, max_weight: float = 1.0,
                   sector_limit=None, target_return=None) -> np.ndarray:
    """给定训练期收益矩阵，求解固定权重向量（供滚动回测复用）。"""
    mu = _expected_returns(ret)
    cov = _robust_cov(ret).values * config.TRADING_DAYS
    names = list(ret.columns)
    n = len(names)
    rf = config.RISK_FREE_RATE

    upper = max_weight
    if max_weight * n < 1 - 1e-9:
        upper = 1.0 / n
    bounds = [(0.0, upper)] * n

    ineq_cons = []
    if sector_limit is not None and sector_limit < 1.0 - 1e-9:
        ineq_cons, _, _ = _build_sector_constraints(names, sector_limit)

    if mode == "equal_weight":
        return np.ones(n) / n
    if mode == "risk_budgeting":
        return _risk_parity_weights(cov, bounds, ineq_cons)
    if mode == "min_variance":
        return _optimize(mu, cov, bounds, rf, lambda w: _portfolio_vol(w, cov),
                         ineq_cons=ineq_cons)
    if mode == "target_return" and target_return is not None:
        tr = float(np.clip(target_return, float(mu.min()), float(mu.max())))
        return _optimize(mu, cov, bounds, rf, lambda w: _portfolio_vol(w, cov),
                         target=tr, ineq_cons=ineq_cons)
    # max_sharpe
    return _optimize(mu, cov, bounds, rf, lambda w: _neg_sharpe(w, mu, cov, rf),
                     ineq_cons=ineq_cons)


def walk_forward_backtest(ret: pd.DataFrame, methods=None,
                          train_window: int = 126, step: int = 20,
                          max_weight: float = 1.0, sector_limit=None,
                          target_return=None, cost_rate: float = config.COST_RATE) -> dict:
    """滚动样本外回测：训练窗滚动推进，评估不同优化方法的样本外表现。

    每过一个 step 交易日，用窗口内最近 train_window 个交易日求权重，
    持有 step 个交易日；样本外区间从首个训练窗结束日起算，加权得到
    各方法的样本外日收益与净值。
    换仓时按 |Δ权重| × cost_rate 扣除调仓成本（首期按满仓建仓扣一次），
    避免忽略换手成本系统性高估样本外表现。
    返回 {中文方法名: {"nav": Series, "stats": dict}}，按夏普降序不可变。
    """
    ret = ret.astype(float).dropna()
    n_assets = ret.shape[1]
    if n_assets < 2 or ret.shape[0] < train_window + step:
        return {}
    methods = methods or ["min_variance", "max_sharpe", "risk_budgeting", "equal_weight"]

    out = {}
    for m in methods:
        port_ret = pd.Series(0.0, index=ret.index, dtype=float)
        cost = pd.Series(0.0, index=ret.index, dtype=float)
        w_prev = None
        for t in range(train_window, ret.shape[0], step):
            train = ret.iloc[t - train_window:t]
            w = _solve_weights(train, m, max_weight, sector_limit, target_return)
            end = min(t + step, ret.shape[0])
            port_ret.iloc[t:end] = ret.iloc[t:end].values @ w
            # 首期按满仓建仓、其后按换仓幅度（|Δw|）计成本；权重归一化后 Σ|w|≈1
            turnover = float(np.abs(w - w_prev).sum()) if w_prev is not None else float(np.abs(w).sum())
            cost.iloc[t] = turnover * cost_rate
            w_prev = w
        oos = (port_ret - cost).iloc[train_window:]
        label = METHOD_LABELS.get(m, m)
        out[label] = {
            "nav": (1 + oos).cumprod(),
            "stats": _returns_stats(oos),
        }
    return out


def walk_forward_summary(wf: dict) -> pd.DataFrame:
    """滚动回测结果汇总为对比表（按夏普降序）。"""
    rows = [{"方法": name, **v["stats"]} for name, v in wf.items()]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("夏普比率", ascending=False)
    return df