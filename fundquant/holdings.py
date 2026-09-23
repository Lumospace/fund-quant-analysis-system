# -*- coding: utf-8 -*-
"""持仓穿透分析模块：把基金前十大重仓股聚合为行业 / 规模风格 / 估值风格暴露。

穿透口径：以每只重仓股的「占净值比例」为权重，向上聚合行业分布与风格暴露，
用于还原基金实际投向的底层结构，而非仅停留在基金名称/类型层面。
"""
import numpy as np
import pandas as pd

from . import data

# 规模风格阈值（总市值，亿元）：>=大盘线为大盘，>=中盘线为中盘，否则小盘
SIZE_LARGE_YI = 800.0
SIZE_MID_YI = 200.0
# 估值风格阈值（市盈率，动态）：<=价值线为价值，>=成长线为成长，中间为平衡；亏损(PE<=0)归高弹性成长
VALUE_PE = 20.0
GROWTH_PE = 40.0


def _classify_size(market_cap_yi: float) -> str:
    if market_cap_yi is None or not np.isfinite(market_cap_yi) or market_cap_yi <= 0:
        return "未知"
    if market_cap_yi >= SIZE_LARGE_YI:
        return "大盘"
    if market_cap_yi >= SIZE_MID_YI:
        return "中盘"
    return "小盘"


def _classify_value(pe) -> str:
    if pe is None or (isinstance(pe, float) and not np.isfinite(pe)):
        return "未知"
    if pe <= 0:
        return "成长"
    if pe <= VALUE_PE:
        return "价值"
    if pe >= GROWTH_PE:
        return "成长"
    return "平衡"


def _normalize(s: pd.Series) -> pd.Series:
    """归一化为占前十大持仓的百分比（合计 100%）。"""
    tot = float(s.sum())
    return (s / tot * 100).round(2) if tot > 0 else s


def stock_lookthrough(holdings: pd.DataFrame) -> dict:
    """对前十大重仓股做行业/风格穿透。

    入参为 data.get_fund_holdings 返回值（至少含「股票代码」「占净值比例」）。
    返回 dict：
      industry     行业      -> 占前十大持仓百分比（Series，降序）
      size         规模风格  -> 占前十大持仓百分比
      value        估值风格  -> 占前十大持仓百分比
      details      逐股明细 DataFrame（行业/总市值亿/PE-TTM/规模风格/估值风格）
      concentration 前十大合计占净值比例（%）
    """
    empty = {"industry": pd.Series(dtype=float), "size": pd.Series(dtype=float),
             "value": pd.Series(dtype=float), "details": pd.DataFrame(), "concentration": 0.0}
    if holdings is None or holdings.empty or "股票代码" not in holdings.columns:
        return empty

    dh = holdings.copy()
    dh["_w"] = pd.to_numeric(dh.get("占净值比例"), errors="coerce").fillna(0.0)

    recs = [{"code": str(r.get("股票代码", "")).split(".")[0],
             "name": str(r.get("股票名称", "")),
             "w": float(r.get("_w", 0.0))} for _, r in dh.iterrows()]

    # 稳健取数：市值/估值走单次全量快照，行业走低并发申万反向映射（均长缓存），内存查表不逐股联网
    spot = data.get_stock_spot_index()
    ind = data.get_stock_industry_index()
    spot_lookup = {} if spot is None or spot.empty else spot.set_index("code").to_dict("index")
    ind_lookup = {} if ind is None or ind.empty else ind.set_index("code").to_dict("index")

    def _fin(x):
        return float(x) if x is not None and np.isfinite(x) else None

    def _enrich(rec):
        code = rec["code"]
        sp = spot_lookup.get(code) or {}
        industry = str((ind_lookup.get(code) or {}).get("industry", "") or "") or "未知"
        mc_yi = _fin(sp.get("market_cap_yi")) or 0.0
        pe = _fin(sp.get("pe"))
        return {
            "股票代码": code,
            "股票名称": rec["name"],
            "行业": industry,
            "占净值比例%": rec["w"],
            "总市值(亿)": round(mc_yi, 1) if mc_yi > 0 else None,
            "PE(动态)": (round(pe, 2) if pe is not None else None),
            "规模风格": _classify_size(mc_yi if mc_yi > 0 else None),
            "估值风格": _classify_value(pe),
        }

    rows = [_enrich(r) for r in recs]

    details = pd.DataFrame(rows)
    industry = details.groupby("行业")["占净值比例%"].sum().sort_values(ascending=False)
    size = (details.groupby("规模风格")["占净值比例%"].sum()
            .reindex(["大盘", "中盘", "小盘", "未知"]).dropna())
    value = details.groupby("估值风格")["占净值比例%"].sum()
    return {
        "industry": _normalize(industry),
        "size": _normalize(size),
        "value": _normalize(value),
        "details": details,
        "concentration": round(float(details["占净值比例%"].sum()), 2),
    }