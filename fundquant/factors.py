# -*- coding: utf-8 -*-
"""量化因子特征工程层（建模核心基底）。

双因子体系：人工核心因子打底 + AI 增量因子辅助。
内置因子自动化校验机制：计算时间序列 IC / IR / 多头超额胜率 / 单调性，
连续失效自动预警并下线，支持人工保留。
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from . import indicators as ind


# ==================== 因子定义 ====================

def _rolling_return(prices, win):
    return prices.pct_change(win)


def _rolling_vol(prices, win=20):
    return prices.pct_change().rolling(win).std() * np.sqrt(252)


def _rolling_sharpe(prices, win=60):
    r = prices.pct_change()
    return (r.rolling(win).mean() / r.rolling(win).std().replace(0, np.nan)) * np.sqrt(252)


class FactorDef:
    def __init__(self, key, name, category, higher_better, desc):
        self.key = key
        self.name = name
        self.category = category
        self.higher_better = higher_better
        self.desc = desc

    def compute(self, prices: pd.Series, hist: pd.DataFrame,
                pe_hist: pd.DataFrame = None) -> pd.Series:
        raise NotImplementedError


class MomentumFactor(FactorDef):
    def __init__(self):
        super().__init__("momentum", "动量因子", "基础", True,
                         "过去 N 日涨跌幅，动量效应，强者恒强")

    def compute(self, prices, hist, pe_hist=None):
        return _rolling_return(prices, 20)


class MeanReversionFactor(FactorDef):
    def __init__(self):
        super().__init__("mean_reversion", "均值回归因子", "基础", True,
                         "价格相对 60 日均线的偏离，负向偏离越大未来反弹概率越高")

    def compute(self, prices, hist, pe_hist=None):
        ma = prices.rolling(60).mean()
        return -(prices / ma - 1)  # 偏离越大（越超跌）因子值越高


class VolatilityFactor(FactorDef):
    def __init__(self):
        super().__init__("volatility", "波动率因子", "基础", False,
                         "低波动因子，历史波动越低稳定性越好")

    def compute(self, prices, hist, pe_hist=None):
        return _rolling_vol(prices, 20)


class ValuationFactor(FactorDef):
    def __init__(self):
        super().__init__("valuation", "估值因子", "基础", True,
                         "真实 PE-TTM 历史分位取反：估值分位越低（越便宜）因子值越高，未来空间越大"
                         "；无真实 PE 数据时回退为价格相对位置")

    def compute(self, prices, hist, pe_hist=None):
        pct = ind.pe_percentile_aligned(prices, pe_hist)
        if pct.empty:
            # 向量化窗口分位替代 O(n·w) 的 rolling.apply(lambda)：rank(pct=True) 含自身，
            # 减 1/n 重建「严格小于」口径，与 PE 分位一致且大幅提速
            w = prices.rolling(252).rank(pct=True) - 1.0 / 252
            return -w  # 兜底：价格越接近一年低位因子值越高
        return -pct  # PE 分位越低（越便宜）因子值越高


class WinRateFactor(FactorDef):
    def __init__(self):
        super().__init__("win_rate", "胜率因子", "基础", True,
                         "过去 60 日上涨交易日占比，胜率高者更稳健")

    def compute(self, prices, hist, pe_hist=None):
        r = prices.pct_change()
        return r.gt(0).rolling(60).mean()


class DrawdownFactor(FactorDef):
    def __init__(self):
        super().__init__("drawdown", "回撤控制因子", "基础", True,
                         "当前较历史高点的回撤（负值），回撤越浅越好")

    def compute(self, prices, hist, pe_hist=None):
        dd = prices / prices.cummax() - 1
        return dd  # 越接近 0 越好


class TrendFactor(FactorDef):
    def __init__(self):
        super().__init__("trend", "时序趋势因子", "AI", True,
                         "短中期均线多头排列强度，刻画趋势方向与强度")

    def compute(self, prices, hist, pe_hist=None):
        ma5 = prices.rolling(5).mean()
        ma20 = prices.rolling(20).mean()
        ma60 = prices.rolling(60).mean()
        return (ma5 - ma20) / (ma20.replace(0, np.nan)) + (ma20 - ma60) / (ma60.replace(0, np.nan))


class TurnAnomalyFactor(FactorDef):
    def __init__(self):
        super().__init__("turn_anomaly", "行情拐点因子", "AI", True,
                         "RSI 超卖程度，超卖越深越接近拐点")

    def compute(self, prices, hist, pe_hist=None):
        r = ind.rsi(prices, 14)
        return 50 - r  # RSI 越低（超卖）因子值越高


class MoneyFlowFactor(FactorDef):
    def __init__(self):
        super().__init__("money_flow", "资金情绪因子", "AI", True,
                         "成交额趋势（量能放大代表资金关注度上升）")

    def compute(self, prices, hist, pe_hist=None):
        if "amount" not in hist.columns:
            return pd.Series(np.nan, index=prices.index)
        amt = pd.to_numeric(hist.set_index("date")["amount"], errors="coerce").reindex(prices.index)
        return amt.pct_change(5)


# ==================== 扩展因子：换手率 ====================

class TurnoverFactor(FactorDef):
    def __init__(self):
        super().__init__("turnover", "换手率因子", "基础", False,
                         "换手率越低，筹码锁定越好，趋势延续概率更高")

    def compute(self, prices, hist, pe_hist=None):
        if "turnover" not in hist.columns:
            return pd.Series(np.nan, index=prices.index)
        t = hist.set_index("date")["turnover"]
        t = pd.to_numeric(t, errors="coerce").reindex(prices.index)
        return -t.rolling(20).mean()  # 低换手正向


class EarningsYieldFactor(FactorDef):
    def __init__(self):
        super().__init__("earnings_yield", "盈利收益率因子", "基础", True,
                         "盈利收益率=1/PE，反映估值高低；值越高代表越被低估，"
                         "长期收益潜力越大")

    def compute(self, prices, hist, pe_hist=None):
        # 盈利收益率 E/P = 1/PE，作为估值代理（仅有 PE 数据，无法还原真实
        # ROE=净利润/净资产，故不把 1/PE 误标为 ROE），滚动均值平滑噪音。
        if pe_hist is None or pe_hist.empty or "pe" not in pe_hist.columns:
            return pd.Series(np.nan, index=prices.index)
        pe = pe_hist.set_index("date")["pe"]
        pe = pd.to_numeric(pe, errors="coerce").dropna().reindex(prices.index, method="ffill")
        ey = 1.0 / pe.replace(0, np.nan)
        return ey.rolling(60).mean()


class CrowdingFactor(FactorDef):
    def __init__(self):
        super().__init__("crowding", "拥挤度因子", "AI", False,
                         "成交额与波动率同步放大的程度，衡量市场拥挤/踩踏风险；"
                         "高拥挤时未来回撤概率上升")

    def compute(self, prices, hist, pe_hist=None):
        r = prices.pct_change()
        vol = r.rolling(20).std() * np.sqrt(252)
        vol_z = (vol - vol.rolling(60).mean()) / vol.rolling(60).std().replace(0, np.nan)

        if "amount" in hist.columns:
            amt = pd.to_numeric(hist.set_index("date")["amount"],
                                errors="coerce").reindex(prices.index)
            amt_ma = amt.rolling(20).mean()
            amt_z = (amt_ma - amt_ma.rolling(60).mean()) / amt_ma.rolling(60).std().replace(0, np.nan)
            crowding = (vol_z.fillna(0) + amt_z.fillna(0)) / 2
        else:
            crowding = vol_z.fillna(0)
        return crowding  # 高拥挤 → 高回撤风险


# ==================== 正交化与中性化 ====================

def orthogonalize(target: pd.Series, X: pd.DataFrame) -> pd.Series:
    """正交化：用线性回归剥离 target 中能被 X 解释的成分，返回残差。

    适用于因子剥离风格暴露（如剥离市值、波动率影响），
    使新因子提供增量信息而非已有因子的冗余表达。
    """
    mask = target.notna() & X.notna().all(axis=1)
    if mask.sum() < 30:
        return target.copy()
    y = target[mask].values.astype(float)
    xv = X[mask].values.astype(float)
    try:
        model = LinearRegression().fit(xv, y)
        resid = y - model.predict(xv)
        out = target.copy()
        out.loc[mask] = resid
        return out
    except Exception:
        return target.copy()


def neutralize(target: pd.Series, benchmark_ret: pd.Series = None) -> pd.Series:
    """市场中性化：剔除基准收益对因子值的影响（回归残差）。

    当 benchmark_ret 为 None 时，使用目标序列自身的滚动均值做 demean。
    中性化后的因子值的市场整体方向无关，仅反映相对强弱。
    """
    if benchmark_ret is not None:
        combined = pd.concat([target.rename("target"),
                              benchmark_ret.rename("bm")], axis=1).dropna()
        if len(combined) < 30:
            return target.copy()
        try:
            model = LinearRegression().fit(combined[["bm"]].values,
                                           combined["target"].values)
            resid = combined["target"].values - model.predict(combined[["bm"]].values)
            out = target.copy()
            out.loc[combined.index] = resid
            return out
        except Exception:
            return target.copy()
    demeaned = target - target.expanding().mean()
    return demeaned


def orthogonalize_factor_frame(ff: pd.DataFrame,
                               against: list = None) -> pd.DataFrame:
    """对因子矩阵做批量正交化：依次将每个因子对 against 中列回归取残差。

    against 为因子名列表（默认所有非自身因子），
    适用于多因子共线性处理，使每个因子提供增量信息。
    """
    out = ff.copy()
    if against is None:
        against = list(ff.columns)
    for col in ff.columns:
        others = [c for c in against if c != col and c in ff.columns]
        if not others:
            continue
        out[col] = orthogonalize(ff[col], ff[others])
    return out


# 因子注册表（人工核心 + AI 增量）
FACTORS: dict = {
    f.key: f for f in [
        MomentumFactor(), MeanReversionFactor(), VolatilityFactor(),
        ValuationFactor(), WinRateFactor(), DrawdownFactor(),
        TrendFactor(), TurnAnomalyFactor(), MoneyFlowFactor(),
        TurnoverFactor(), EarningsYieldFactor(), CrowdingFactor(),
    ]
}


def compute_factor_frame(prices: pd.Series, hist: pd.DataFrame,
                         pe_hist: pd.DataFrame = None) -> pd.DataFrame:
    """计算全部因子值（pe_hist 供估值因子使用），返回与 prices 对齐的 DataFrame。"""
    out = pd.DataFrame(index=prices.index)
    for key, f in FACTORS.items():
        try:
            out[key] = f.compute(prices, hist, pe_hist)
        except Exception:
            out[key] = np.nan
    return out


# ==================== 因子有效性校验 ====================

def _spearman(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    m = (~a.isna()) & (~b.isna())
    if m.sum() < 20:
        return np.nan
    with np.errstate(all="ignore"):
        return stats.spearmanr(a[m].astype(float), b[m].astype(float)).statistic


def factor_ic(factor: pd.Series, forward_ret: pd.Series) -> float:
    """时间序列 IC（因子值与未来收益的 Spearman 相关系数）。"""
    return _spearman(factor, forward_ret)


def factor_ir(ic_series: pd.Series) -> float:
    ic_series = ic_series.dropna()
    if len(ic_series) < 2:
        return 0.0
    std = float(ic_series.std(ddof=1))
    if not np.isfinite(std) or std == 0:
        return 0.0
    with np.errstate(all="ignore"):
        return float(ic_series.mean() / std)


def factor_win_rate(factor: pd.Series, forward_ret: pd.Series, top_q: float = 0.7) -> float:
    """多头超额胜率：因子进入高分位时，未来收益为正的比例 - 全样本正收益比例。"""
    f = pd.to_numeric(factor, errors="coerce")
    fr = pd.to_numeric(forward_ret, errors="coerce")
    m = (~f.isna()) & (~fr.isna())
    if m.sum() < 30:
        return np.nan
    f, fr = f[m], fr[m]
    thr = f.quantile(top_q)
    top = fr[f >= thr]
    base = float((fr > 0).mean())
    return float((top > 0).mean() - base)


def factor_monotonicity(factor: pd.Series, forward_ret: pd.Series, n_group: int = 5) -> float:
    """单调性：分组均值与组序号的 Spearman 相关（接近 ±1 表示单调）。"""
    f = pd.to_numeric(factor, errors="coerce")
    fr = pd.to_numeric(forward_ret, errors="coerce")
    m = (~f.isna()) & (~fr.isna())
    if m.sum() < n_group * 8:
        return np.nan
    f, fr = f[m], fr[m]
    try:
        grp = pd.qcut(f.rank(method="first"), n_group, labels=False, duplicates="drop")
    except ValueError:
        return np.nan
    mean_ret = pd.Series(fr.values).groupby(grp).mean()
    if len(mean_ret) < 3:
        return np.nan
    return _spearman(pd.Series(range(len(mean_ret))), mean_ret)


def validate_factors(prices: pd.Series, hist: pd.DataFrame,
                     forward: int = 20, ic_threshold: float = 0.02,
                     down_months: int = 3, pe_hist: pd.DataFrame = None,
                     ff: pd.DataFrame = None) -> pd.DataFrame:
    """因子自动校验：返回每个因子的 IC/IR/胜率/单调性 与 状态。

    连续 down_months 个月 IC 绝对值低于阈值 -> 自动下线（状态=已下线）。
    传入 ff 可复用已计算的因子矩阵，避免多次重复滚动计算。
    """
    if ff is None:
        ff = compute_factor_frame(prices, hist, pe_hist)
    fwd = prices.pct_change(forward).shift(-forward)  # 未来 forward 日收益

    rows = []
    for key, f in FACTORS.items():
        series = ff[key]
        ic_full = factor_ic(series, fwd)
        muniv = factor_monotonicity(series, fwd)
        win = factor_win_rate(series, fwd)

        # 月度滚动 IC（按月升序），用于失效占比判定
        month = pd.to_datetime(series.index).to_period("M")
        tmp = pd.DataFrame({"f": series.values, "r": fwd.values, "m": month.values})
        mics = {}
        for m, g in tmp.dropna().groupby("m"):
            if len(g) >= 10:
                with np.errstate(all="ignore"):
                    try:
                        mics[m] = stats.spearmanr(g["f"], g["r"]).statistic
                    except Exception:
                        mics[m] = np.nan
        monthly_ic = pd.Series(mics, dtype=float).sort_index()  # 按月升序

        ic_series = monthly_ic.dropna()
        n_valid = len(ic_series)
        ir = factor_ir(ic_series) if n_valid > 1 else np.nan
        # 月度 IC 均值的 t 统计量：t = mean/std × sqrt(n)（检验「因子方向无关的 IC 均值 = 0」）
        t_val = ir * np.sqrt(n_valid) if n_valid > 1 and np.isfinite(ir) else np.nan

        # 累计连续失效月数（仅作展示信息，不再作为下线判据）
        consec = 0
        max_consec = 0
        for v in monthly_ic.tolist():
            if np.isnan(v) or abs(v) < ic_threshold:
                consec += 1
            else:
                consec = 0
            max_consec = max(max_consec, consec)

        # 下线/预警判定：近 window 个月滚动失效占比（替代机械连续计数）
        # 旧式「连续 down_months 月失效即下线」在短历史上极易误杀新成立基金，
        # 且忽略因子在不同市场状态下的周期性失效。改用滚动窗口失效占比 +
        # 最小样本保护：样本不足时不轻易下线。
        window = max(down_months * 2, 12)
        recent = ic_series.tail(window)  # 最近 window 个有效月
        fail_ratio = float((recent.abs() < ic_threshold).sum() / window) if n_valid else 0.0

        if n_valid < down_months:
            status = "样本不足"
        elif fail_ratio >= 0.6:
            status = "已下线"
        elif fail_ratio >= 0.4:
            status = "预警"
        else:
            status = "有效"

        rows.append({
            "因子": f.name, "类别": f.category, "方向": "正向" if f.higher_better else "负向",
            "IC": round(ic_full, 4) if not np.isnan(ic_full) else np.nan,
            "IR": round(ir, 3) if not np.isnan(ir) else np.nan,
            "t值": round(t_val, 2) if not np.isnan(t_val) else np.nan,
            "多头超额胜率": round(win, 4) if not np.isnan(win) else np.nan,
            "单调性": round(muniv, 3) if not np.isnan(muniv) else np.nan,
            "连续失效月数": max_consec, "状态": status, "说明": f.desc,
        })
    df = pd.DataFrame(rows).sort_values("IC", key=lambda s: s.abs(), ascending=False)
    return df


def ic_time_series(prices: pd.Series, hist: pd.DataFrame, forward: int = 20,
                   window: int = 60, step: int = 10, pe_hist: pd.DataFrame = None,
                   ff: pd.DataFrame = None):
    """因子滚动 IC 时序 + 累计 IC 曲线。

    对每个因子，在滚动 window 交易日内计算因子值与未来 forward 日收益的
    Spearman IC，每 step 个交易日采样一点构成 IC 时序；逐点累加得累计 IC，
    累计 IC 持续上行代表该因子方向长期稳定有效。
    返回 (ic_frame, cum_ic_frame)：index 为采样日期，columns 为因子中文名。
    """
    if ff is None:
        ff = compute_factor_frame(prices, hist, pe_hist)
    fwd = prices.pct_change(forward).shift(-forward)
    name_map = {k: f.name for k, f in FACTORS.items()}
    idx = list(ff.index)
    n = len(idx)
    rows = []
    for start in range(0, n - window + 1, step):
        end = start + window
        seg_f = ff.iloc[start:end]
        seg_r = fwd.iloc[start:end]
        row = {"date": idx[end - 1]}
        for key in FACTORS.keys():
            row[key] = factor_ic(seg_f[key], seg_r)
        rows.append(row)
    ic_frame = pd.DataFrame(rows)
    if ic_frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    ic_frame = ic_frame.set_index("date")
    ic_frame.columns = [name_map.get(c, c) for c in ic_frame.columns]
    return ic_frame, ic_frame.cumsum()


def effective_factor_keys(prices: pd.Series, hist: pd.DataFrame,
                          forward: int = 20, pe_hist: pd.DataFrame = None) -> list:
    """返回当前有效因子 key 列表（用于预测特征筛选）。"""
    v = validate_factors(prices, hist, forward, pe_hist=pe_hist)
    if v.empty:
        return list(FACTORS.keys())
    ok = v[v["状态"] != "已下线"]["因子"]
    name2key = {f.name: k for k, f in FACTORS.items()}
    return [name2key[n] for n in ok if n in name2key]


def factor_correlation_matrix(prices: pd.Series, hist: pd.DataFrame,
                               pe_hist: pd.DataFrame = None,
                               ff: pd.DataFrame = None) -> pd.DataFrame:
    """计算各因子间的 Spearman 相关系数矩阵，用于发现冗余因子对。

    返回 DataFrame，index/columns 均为因子中文名。
    """
    if ff is None:
        ff = compute_factor_frame(prices, hist, pe_hist)
    if ff.empty or ff.shape[1] < 2:
        return pd.DataFrame()
    name_map = {k: f.name for k, f in FACTORS.items()}
    ff = ff.rename(columns=name_map)
    return ff.corr(method="spearman")


def high_correlation_warning(corr: pd.DataFrame, threshold: float = 0.8) -> list:
    """检查相关系数矩阵，返回超过 threshold 的高相关因子对列表。

    每项为 (因子A, 因子B, 相关系数)，按相关系数绝对值降序排列。
    """
    if corr is None or corr.empty:
        return []
    pairs = []
    names = corr.columns.tolist()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v = corr.iloc[i, j]
            if not np.isnan(v) and abs(v) >= threshold:
                pairs.append((names[i], names[j], round(v, 3)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs