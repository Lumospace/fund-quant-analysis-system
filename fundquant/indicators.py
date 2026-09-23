# -*- coding: utf-8 -*-
"""指标计算层：收益、风险、估值、技术指标。全部基于价格序列，纯函数无副作用。"""
import numpy as np
import pandas as pd

import config


def to_price_series(hist: pd.DataFrame, col: str = "close") -> pd.Series:
    """从历史行情抽取价格序列（按日期索引）。支持 close / nav 列。"""
    if hist is None or hist.empty:
        return pd.Series(dtype=float)
    df = hist.copy()
    if col in df.columns:
        s = df[col]
    elif "nav" in df.columns:
        s = df["nav"]
    else:
        return pd.Series(dtype=float)
    s = pd.to_numeric(s, errors="coerce").dropna()
    idx = pd.to_datetime(df.loc[s.index, "date"]).values
    return pd.Series(s.values, index=idx)


def daily_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def cumulative_return(prices: pd.Series) -> float:
    if len(prices) < 2:
        return 0.0
    return float(prices.iloc[-1] / prices.iloc[0] - 1)


def period_return(prices: pd.Series, days: int) -> float:
    """近 days 个交易日收益。"""
    if len(prices) <= days:
        return cumulative_return(prices)
    return float(prices.iloc[-1] / prices.iloc[-(days + 1)] - 1)


def annualized_return(prices: pd.Series) -> float:
    if len(prices) < 2:
        return 0.0
    total = cumulative_return(prices)
    years = (len(prices) - 1) / config.TRADING_DAYS
    if years <= 0:
        return 0.0
    if total <= -1:
        return -1.0
    return float((1 + total) ** (1 / years) - 1)


def annualized_volatility(prices: pd.Series) -> float:
    r = daily_returns(prices)
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(config.TRADING_DAYS))


def downside_deviation(prices: pd.Series, mar: float = 0.0) -> float:
    r = daily_returns(prices)
    if len(r) == 0:
        return 0.0
    downside = r[r < mar]
    if len(downside) == 0:
        return 0.0
    return float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(config.TRADING_DAYS))


def drawdown_series(prices: pd.Series) -> pd.Series:
    return prices / prices.cummax() - 1


def max_drawdown(prices: pd.Series) -> float:
    dd = drawdown_series(prices)
    if len(dd) == 0:
        return 0.0
    return float(dd.min())


def longest_drawdown_days(prices: pd.Series) -> int:
    """最长回撤修复周期（连续处于水下/未创新高的交易日数）。"""
    dd = drawdown_series(prices)
    if len(dd) == 0:
        return 0
    m = 0
    cur = 0
    for v in dd:
        if v < 0:
            cur += 1
            m = max(m, cur)
        else:
            cur = 0
    return int(m)


def sharpe_ratio(prices: pd.Series, rf: float = None) -> float:
    rf = rf if rf is not None else config.RISK_FREE_RATE
    r = daily_returns(prices)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    excess = r.mean() * config.TRADING_DAYS - rf
    return float(excess / (r.std(ddof=1) * np.sqrt(config.TRADING_DAYS)))


def sortino_ratio(prices: pd.Series, rf: float = None) -> float:
    rf = rf if rf is not None else config.RISK_FREE_RATE
    r = daily_returns(prices)
    if len(r) < 2:
        return 0.0
    dd = downside_deviation(prices)
    if dd <= 0:
        return 0.0
    excess = r.mean() * config.TRADING_DAYS - rf
    return float(excess / dd)


def calmar_ratio(prices: pd.Series) -> float:
    mdd = abs(max_drawdown(prices))
    if mdd == 0:
        return 0.0
    return float(annualized_return(prices) / mdd)


def var_historical(prices: pd.Series, alpha: float = 0.95) -> float:
    """历史模拟法 VaR（返回正数，表示亏损幅度）。"""
    r = daily_returns(prices)
    if len(r) < 10:
        return 0.0
    return float(-np.percentile(r, (1 - alpha) * 100))


def cvar_historical(prices: pd.Series, alpha: float = 0.95) -> float:
    r = daily_returns(prices)
    if len(r) < 10:
        return 0.0
    # CVaR（期望损失）= 最差 (1-alpha) 分位内收益的均值；用分位数本身（负值）筛选尾部
    tail = r[r <= np.percentile(r, (1 - alpha) * 100)]
    if len(tail) == 0:
        return 0.0
    return float(-tail.mean())


def mc_tail_from_returns(r, alpha: float = 0.95, horizon: int = 1,
                         n_paths: int = 20000, seed: int = 2024) -> tuple:
    """参数化蒙特卡洛尾部损失（基于日收益序列，几何布朗运动假设）。

    返回 (VaR, CVaR)，均为正数、表示 horizon 个交易日的亏损幅度。
    相比历史分位法更稳健：短样本或涨跌集中时历史分位可能失真，MC 用分布假设
    平滑估计尾部；固定 seed 保证跨运行可复现。
    """
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) < 10:
        return 0.0, 0.0
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    if sigma <= 0:
        sigma = 1e-4
    rng = np.random.default_rng(seed)
    # 几何布朗运动：horizon 期累计收益（对数收益经 -0.5σ² 调整为几何漂移）
    log_ret = (mu - 0.5 * sigma ** 2) * horizon \
        + sigma * np.sqrt(horizon) * rng.standard_normal(n_paths)
    sim_ret = np.exp(log_ret) - 1.0
    q = float(np.percentile(sim_ret, (1 - alpha) * 100))
    var = float(-q)
    tail = sim_ret[sim_ret <= q]
    cvar = float(-tail.mean()) if len(tail) else var
    return var, cvar


def var_monte_carlo(prices: pd.Series, alpha: float = 0.95, horizon: int = 1,
                    n_paths: int = 20000, seed: int = 2024) -> float:
    """蒙特卡洛 VaR（输入价格序列，内部取日收益；返回正数亏损幅度）。"""
    return mc_tail_from_returns(daily_returns(prices), alpha, horizon, n_paths, seed)[0]


def cvar_monte_carlo(prices: pd.Series, alpha: float = 0.95, horizon: int = 1,
                     n_paths: int = 20000, seed: int = 2024) -> float:
    """蒙特卡洛 CVaR（输入价格序列；返回正数期望亏损幅度）。"""
    return mc_tail_from_returns(daily_returns(prices), alpha, horizon, n_paths, seed)[1]


# ==================== 技术指标 ====================

def moving_average(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window).mean()


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def bollinger(prices: pd.Series, window: int = 20, k: float = 2.0):
    mid = moving_average(prices, window)
    std = prices.rolling(window).std()
    return mid, mid + k * std, mid - k * std


def valuation_percentile_from_pe(pe_hist) -> float:
    """真实估值分位：当前 PE-TTM 在历史 PE-TTM 序列中的分位（0-1，越高越贵）。

    与旧的价格分位不同，此处基于真实盈利估值指标（PE-TTM）而非净值/价格。
    无充足真实 PE 数据时返回 NaN（表示"数据不足"，而非用价格分位冒充）。
    """
    if pe_hist is None or getattr(pe_hist, "empty", True) or "pe" not in pe_hist.columns:
        return np.nan
    pe = pd.to_numeric(pe_hist["pe"], errors="coerce").dropna()
    if len(pe) < 36:
        return np.nan
    cur = pe.iloc[-1]
    return float((pe < cur).mean())


def pe_percentile_series(pe_hist) -> pd.Series:
    """由真实 PE-TTM 历史构造逐期估值分位序列（扩张窗口，严格无未来信息）。

    每个时点的分位 = 当前 PE 在"该时点之前"的历史中的分位（排除自身与未来），
    用于定投回测/时点判断，避免前视偏差。返回按 date 索引、0-1 的 Series。
    """
    if pe_hist is None or getattr(pe_hist, "empty", True):
        return pd.Series(dtype=float)
    if "date" not in pe_hist.columns or "pe" not in pe_hist.columns:
        return pd.Series(dtype=float)
    pdf = pd.DataFrame({
        "date": pd.to_datetime(pe_hist["date"]),
        "pe": pd.to_numeric(pe_hist["pe"], errors="coerce"),
    }).dropna().sort_values("date").reset_index(drop=True)
    if len(pdf) < 36:
        return pd.Series(dtype=float)
    vals = []
    pe_arr = pdf["pe"].values
    for i in range(len(pe_arr)):
        hist = pe_arr[:i]  # 截至前一期的历史，严格无前视
        vals.append(float((hist < pe_arr[i]).mean()) if len(hist) >= 24 else np.nan)
    return pd.Series(vals, index=pdf["date"].values)


def pe_percentile_aligned(prices: pd.Series, pe_hist) -> pd.Series:
    """真实 PE-TTM 分位序列（0-1，扩张窗口无前视），对齐到价格交易日索引。

    返回与 prices 同索引的 Series；无充足 PE 数据或无法对齐时返回空 Series，
    供因子层/预测层取用（缺失时由调用方决定是否回退）。
    """
    pct = pe_percentile_series(pe_hist)
    if pct.empty:
        return pd.Series(dtype=float)
    left = (pd.DataFrame({"date": pd.to_datetime(pd.Series(prices.index))})
            .sort_values("date").reset_index(drop=True))
    right = (pct.rename("vp").reset_index()
             .rename(columns={"index": "date"}).sort_values("date"))
    merged = pd.merge_asof(left, right, on="date", direction="backward")
    return pd.Series(merged["vp"].values, index=pd.to_datetime(merged["date"]))


def valuation_status(percentile) -> str:
    """估值分位 -> 状态标签。None/NaN 表示无真实估值数据。"""
    if percentile is None:
        return "数据不足"
    p = float(percentile)
    if np.isnan(p):
        return "数据不足"
    if p < 0.2:
        return "低估"
    if p < 0.5:
        return "偏低"
    if p < 0.8:
        return "中性"
    return "高估"


# ==================== 综合指标包 ====================

def compute_fund_metrics(hist: pd.DataFrame, col: str = "close",
                         pe_hist: pd.DataFrame = None) -> dict:
    """一次性计算单个基金的核心指标，供评分/报告/展示复用。

    pe_hist: 真实 PE-TTM 历史（date, pe），用于真实估值分位；不传则该指标为 None。
    """
    prices = to_price_series(hist, col)
    if len(prices) < 20:
        return {}
    vp = valuation_percentile_from_pe(pe_hist)
    return {
        "最新净值": round(float(prices.iloc[-1]), 4),
        "累计收益": round(cumulative_return(prices) * 100, 2),
        "年化收益": round(annualized_return(prices) * 100, 2),
        "近1月": round(period_return(prices, 21) * 100, 2),
        "近3月": round(period_return(prices, 63) * 100, 2),
        "近6月": round(period_return(prices, 126) * 100, 2),
        "近1年": round(period_return(prices, 252) * 100, 2),
        "年化波动率": round(annualized_volatility(prices) * 100, 2),
        "最大回撤": round(max_drawdown(prices) * 100, 2),
        "夏普比率": round(sharpe_ratio(prices), 3),
        "索提诺比率": round(sortino_ratio(prices), 3),
        "卡玛比率": round(calmar_ratio(prices), 3),
        "VaR95": round(var_historical(prices) * 100, 2),
        "下行风险": round(downside_deviation(prices) * 100, 2),
        "最长回撤天数": longest_drawdown_days(prices),
        "估值分位": round(vp * 100, 1) if not np.isnan(vp) else None,
        "样本数": len(prices),
    }


def risk_level(vol: float) -> str:
    """年化波动率 -> 风险等级。"""
    if vol < 0.10:
        return "低风险"
    if vol < 0.20:
        return "中风险"
    return "高风险"


def rolling_return(prices: pd.Series, window: int = 60) -> pd.Series:
    """滚动窗口累计收益率（%）：最近 window 个交易日的区间收益。"""
    p = pd.to_numeric(prices, errors="coerce").dropna()
    if len(p) <= window:
        return pd.Series(dtype=float)
    return (p / p.shift(window) - 1) * 100


def rolling_volatility(prices: pd.Series, window: int = 60) -> pd.Series:
    """滚动窗口年化波动率（%）：日收益 rolling 标准差 × √252。"""
    r = pd.to_numeric(prices, errors="coerce").pct_change().dropna()
    if len(r) < window:
        return pd.Series(dtype=float)
    return r.rolling(window).std(ddof=0) * np.sqrt(config.TRADING_DAYS) * 100


def rolling_risk_return_points(prices: pd.Series, window: int = 60, step: int = 20) -> pd.DataFrame:
    """滚动子窗口的（年化波动, 年化收益）采样点，用于收益-风险散点图。

    以 window 为滚动窗、step 为步长滑动采样，每窗计算年化收益与年化波动，
    返回 DataFrame 列：date / 年化收益 / 年化波动（单位 %）。末行为最新窗口。
    """
    p = pd.to_numeric(prices, errors="coerce").dropna()
    vals = p.values
    idx = list(p.index)
    rows = []
    for end in range(window, len(vals) + 1, step):
        seg = vals[end - window:end]
        rets = np.diff(seg) / seg[:-1]
        if len(rets) < 10 or seg[0] <= 0:
            continue
        ann_ret = (seg[-1] / seg[0]) ** (config.TRADING_DAYS / window) - 1
        ann_vol = float(rets.std(ddof=0) * np.sqrt(config.TRADING_DAYS))
        rows.append({"date": idx[end - 1], "年化收益": round(ann_ret * 100, 2),
                     "年化波动": round(ann_vol * 100, 2)})
    return pd.DataFrame(rows)