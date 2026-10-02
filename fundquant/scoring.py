# -*- coding: utf-8 -*-
"""量化评分模型：四维度加权综合评分，全称可解释、分项可溯源。

综合评分 = 收益35% + 风险35% + 估值20% + 稳定性10%
每项细分指标给出 0-100 分与扣分/加分原因。
"""
import numpy as np

import config


def _clamp(x, lo=0.0, hi=100.0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 50.0
    return float(max(lo, min(hi, x)))


def _linear(v, x0, x1, y0=100.0, y1=0.0):
    """递增分段线性映射：v<=x0 -> y0，v>=x1 -> y1，之间线性（x0 < x1）。"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if v <= x0:
        return y0
    if v >= x1:
        return y1
    return y0 + (y1 - y0) * (v - x0) / (x1 - x0)


def _linear_desc(v, x_bad, x_good, y_bad=0.0, y_good=100.0):
    """递减分段线性映射（数值越小越好）：v>=x_bad -> y_bad，v<=x_good -> y_good，之间线性。

    用于波动率、最长回撤天数这类"越小越优"的指标；x_bad 为最差值，x_good 为最优值。
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if v >= x_bad:
        return y_bad
    if v <= x_good:
        return y_good
    return y_bad + (y_good - y_bad) * (x_bad - v) / (x_bad - x_good)


# ==================== 分维度评分 ====================

def score_return(m: dict) -> tuple:
    """收益维度：年化收益 + 近1年收益。亏损区间有梯度，不再给保底分。"""
    ann = m.get("年化收益", np.nan)      # 单位 %
    y1 = m.get("近1年", np.nan)          # 单位 %
    s_ann = _linear(ann, -20, 30, 0, 100)  # -20% ->0分, 30% ->100分
    s_y1 = _linear(y1, -10, 40, 0, 100)
    score = _clamp(0.6 * s_ann + 0.4 * s_y1)
    reasons = [
        f"年化收益 {ann}%（得分 {_clamp(s_ann):.0f}，-20%→0分、30%→100分线性）",
        f"近1年收益 {y1}%（得分 {_clamp(s_y1):.0f}，-10%→0分、40%→100分线性）",
    ]
    return round(score, 1), reasons


def score_risk(m: dict) -> tuple:
    """风险维度：最大回撤 + 波动率 + 夏普（回撤/波动越低越好，夏普越高越好）。"""
    mdd = m.get("最大回撤", np.nan)      # 负值百分比，如 -25
    vol = m.get("年化波动率", np.nan)    # 单位 %
    sharpe = m.get("夏普比率", np.nan)
    s_mdd = _linear(mdd, -50, -5, 0, 100)        # 回撤 -50%以上 0分，-5%以内 100分
    s_vol = _linear_desc(vol, 40, 5, 0, 100)      # 波动 40%以上 0分，5%以内 100分
    s_sharpe = _linear(sharpe, -0.5, 2.0, 0, 100)  # 夏普 -0.5以下 0分，2.0以上 100分
    score = _clamp(0.4 * s_mdd + 0.4 * s_vol + 0.2 * s_sharpe)
    reasons = [
        f"最大回撤 {mdd}%（得分 {_clamp(s_mdd):.0f}，-50%→0分、-5%→100分）",
        f"年化波动率 {vol}%（得分 {_clamp(s_vol):.0f}，40%→0分、5%→100分）",
        f"夏普比率 {sharpe}（得分 {_clamp(s_sharpe):.0f}，-0.5→0分、2.0→100分）",
    ]
    return round(score, 1), reasons


def score_valuation(m: dict) -> tuple:
    """估值维度：真实估值分位越低越好。无真实 PE 数据时按中性处理并显式说明。"""
    q = m.get("估值分位", np.nan)  # 0-100，越低越便宜；None 表示无真实估值数据
    status = m.get("估值状态", "")
    if q is None or (isinstance(q, float) and np.isnan(q)):
        score = 50.0  # 信息缺失：中性，不偏向
        reasons = ["估值分位无真实 PE-TTM 数据（该基金不跟踪可解析的宽基指数），"
                   "估值维度按中性 50 分处理，已不再用价格分位代替"]
        return round(score, 1), reasons
    s = _linear(q, 0, 100, 100, 0)
    score = _clamp(s)
    reasons = [f"估值分位 {q}%（{status}，分位越低得分越高，得分 {score:.0f}）"]
    return round(score, 1), reasons


def score_stability(m: dict) -> tuple:
    """稳定性维度：索提诺 + 最长回撤天数（索提诺越高越好，回撤修复越短越好）。"""
    sortino = m.get("索提诺比率", np.nan)
    ldd = m.get("最长回撤天数", np.nan)
    s_sort = _linear(sortino, -0.5, 2.0, 0, 100)
    s_ldd = _linear_desc(ldd, 750, 30, 0, 100)  # 750天以上 0分，30天以内 100分
    score = _clamp(0.6 * s_sort + 0.4 * s_ldd)
    reasons = [
        f"索提诺比率 {sortino}（得分 {_clamp(s_sort):.0f}，-0.5→0分、2.0→100分）",
        f"最长回撤修复周期 {ldd} 天（得分 {_clamp(s_ldd):.0f}，750天→0分、30天→100分）",
    ]
    return round(score, 1), reasons


# ==================== 综合评分 ====================

def composite_score(m: dict, style: str = "稳健") -> dict:
    """四维度加权综合评分，返回分项 + 总分 + 评级 + 原因。

    风格修正：保守放大风险惩罚、进取减弱（见 STYLE_PROFILES.score_penalty_risk）。
    """
    dims = {
        "收益": score_return(m),
        "风险": score_risk(m),
        "估值": score_valuation(m),
        "稳定性": score_stability(m),
    }

    weights = dict(config.SCORE_WEIGHTS)
    pen = config.STYLE_PROFILES.get(style, config.STYLE_PROFILES["稳健"])["score_penalty_risk"]

    # 风格修正：对风险维度做加权调整，其它维按比例归一
    risk_base = weights["风险"]
    # 惩罚系数作用于风险得分：pen>1 时风险得分被压低
    adjusted = {}
    for k, (s, r) in dims.items():
        if k == "风险":
            penalty = (100 - s) * (pen - 1.0)
            adjusted[k] = _clamp(s - penalty)
        else:
            adjusted[k] = s

    total = sum(adjusted[k] * weights[k] for k in weights)
    breakdown = {
        k: {"score": adjusted[k], "weight": weights[k], "reasons": dims[k][1]}
        for k in weights
    }

    rating = _rating(total)
    return {
        "total": round(total, 1),
        "rating": rating,
        "style": style,
        "breakdown": breakdown,
    }


def _rating(total: float) -> str:
    if total >= 80:
        return "优秀"
    if total >= 70:
        return "良好"
    if total >= 60:
        return "中等"
    if total >= 45:
        return "一般"
    return "较差"


# ==================== 跨基金优选评分（用于筛选模块） ====================

def score_universe(rank_df: pd.DataFrame) -> pd.DataFrame:
    """基于开放基金排行数据做跨截面评分（仅收益维度，风险数据需补历史）。

    使用 近1月/3月/6月/1年 + 今年来 分位排名合成 0-100 分。
    """
    df = rank_df.copy()
    cols = [c for c in ["m1", "m3", "m6", "y1", "ytd"] if c in df.columns]
    if not cols:
        df["综合评分"] = np.nan
        return df
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 分位排名（百分位，越高越好），缺失填 0.5
    rank_pct = df[cols].rank(pct=True)
    rank_pct = rank_pct.fillna(0.5)
    w = {"m1": 0.15, "m3": 0.2, "m6": 0.2, "y1": 0.3, "ytd": 0.15}
    ws = sum(w[c] for c in cols)
    df["综合评分"] = round(
        sum(rank_pct[c] * w[c] for c in cols if c in w) / ws * 100, 1)
    df["评级"] = df["综合评分"].map(_rating)
    df = df.sort_values("综合评分", ascending=False).reset_index(drop=True)
    return df


# ==================== 白话释义映射 ====================

GLOSSARY = {
    "夏普比率": "每承担一单位波动能换来多少超额收益，越高越划算，一般 >1 算不错。",
    "索提诺比率": "只惩罚下行波动的夏普，越能抗跌越好。",
    "卡玛比率": "年化收益 ÷ 最大回撤，衡量“每跌 1% 能赚多少”，越高越值。",
    "最大回撤": "历史最高点跌到最低点的最大跌幅，衡量最倒霉时最多亏多少。",
    "估值分位": "当前指数 PE(TTM) 在其历史区间中的分位，越低代表估值越便宜（安全边际越高）。",
    "VaR(95%)": "95% 置信度下单日可能的最大亏损幅度，越小越安全。",
    "IC": "因子值与未来收益的相关性，绝对值越大因子预测力越强。",
    "IR": "IC 的均值 ÷ 波动，衡量因子稳定性，越高越可靠。",
}