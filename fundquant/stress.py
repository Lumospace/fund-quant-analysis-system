# -*- coding: utf-8 -*-
"""极端行情专项压力测试模块（可靠性保障）。

内置 6 类历史行情切片（2015 股灾 / 2018 去杠杆 / 2020 疫情 / 2020 后牛市 /
2022 熊市 / 2024 量化冲击），并支持参数化蒙特卡洛压力模拟。
对无相关历史区间的基金/组合，按各历史时期的剧烈程度（波动放大倍数 + 时长）做
蒙特卡洛兜底；同时提供独立 what-if 情景。测算最大回撤、资产缩水幅度、修复周期、波动率。
"""
import numpy as np
import pandas as pd

import config


def _scenario_returns(prices_dict: dict, start: str, end: str) -> pd.DataFrame:
    """抽取情景区间内各资产的日收益，并保证日期对齐。"""
    rets = {}
    for name, s in prices_dict.items():
        s = pd.to_numeric(s, errors="coerce").dropna()
        s.index = pd.to_datetime(s.index)
        s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
        if len(s) > 10:
            rets[name] = s.pct_change().dropna()
    if not rets:
        return pd.DataFrame()
    df = pd.DataFrame(rets)
    return df


def _full_returns(prices_dict: dict) -> pd.DataFrame:
    """全区间对齐的收益矩阵（用于蒙特卡洛兜底，不依赖具体历史切片）。"""
    closes = {}
    for name, s in prices_dict.items():
        s = pd.to_numeric(s, errors="coerce").dropna()
        s.index = pd.to_datetime(s.index)
        if len(s) > 5:
            closes[name] = s
    if not closes:
        return pd.DataFrame()
    px = pd.DataFrame(closes).dropna()
    return px.pct_change().dropna()


def _portfolio_returns(ret_df: pd.DataFrame, weights: dict) -> pd.Series:
    """按权重合成组合的收益序列（权重自动归一化）。"""
    if ret_df is None or ret_df.empty:
        return pd.Series(dtype=float)
    w = np.array([weights.get(c, 0.0) for c in ret_df.columns], dtype=float)
    if w.sum() > 0:
        w = w / w.sum()
    else:
        w = np.ones(len(ret_df.columns)) / len(ret_df.columns)
    return pd.Series(ret_df.values @ w, index=ret_df.index).dropna()


def _nav_from_ret(ret: pd.Series) -> pd.Series:
    return (1 + ret).cumprod()


def _walk_stats(nav: pd.Series) -> dict:
    """单资产/组合在情景内的关键指标。"""
    nav = nav.dropna()
    if len(nav) < 2:
        return {}
    peak = nav.cummax()
    dd = nav / peak - 1
    mdd = float(dd.min())
    # 资产缩水幅度：期末相对期初的涨跌幅（正=扩张、负=缩水）
    shrink = float(nav.iloc[-1] / nav.iloc[0] - 1)
    # 修复周期：从最深回撤点恢复到前高的交易日数（若未修复，计"未修复"）
    trough_pos = int(np.argmin(dd.values))
    peak_val = float(nav.iloc[: trough_pos + 1].max())
    recover_days = None
    recovered = False
    days = 0
    for v in nav.iloc[trough_pos + 1:].values:
        days += 1
        if v >= peak_val:
            recovered = True
            recover_days = days
            break
    vol = float(nav.pct_change().std(ddof=1) * np.sqrt(config.TRADING_DAYS))
    # 修复周期：已修复给出天数；未修复时附"期末距区间高点"的距离，
    # 避免各下跌情景一律显示"未修复"而看不出差异。
    if recovered:
        recover_text = recover_days
    else:
        end_dd = float(nav.iloc[-1] / float(nav.max()) - 1)
        recover_text = f"未修复(期末距高点{end_dd * 100:.1f}%)"
    return {
        "最大回撤": round(mdd * 100, 2),
        "缩水幅度": round(shrink * 100, 2),
        "年化波动率": round(vol * 100, 2),
        "修复周期(交易日)": recover_text,
    }


def _stable_seed(name: str) -> int:
    """由情景名生成稳定随机种子（跨运行可复现，且不同情景种子不同）。"""
    import hashlib
    return int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % (2**32)


def _monte_carlo_pressure(returns, stress_mult: float = 2.5, days: int = 60,
                          n_paths: int = 300, drift_sign: float = -1.0,
                          seed: int = 2024) -> dict:
    """参数化蒙特卡洛压力模拟（几何布朗运动）。

    用收益序列自身的历史波动率 sigma，叠加压力漂移（drift_sign 控制上行/下行）
    与波动放大 stress_mult，生成 n_paths 条合成压力路径，返回每条路径最大回撤/
    期末缩水的悲观分位（10% 分位）。用于历史切片无法覆盖的新基金兜底，
    也作为独立的 what-if 压力情景。seed 由情景名生成，保证不同情景结果各异。
    """
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 20:
        return {}
    sigma = float(r.std())
    if sigma <= 0:
        sigma = 0.01
    sigma_s = sigma * stress_mult
    # 漂移保持极小（仅体现牛/熊方向），避免持续单边漂移导致"必然清零"式极端回撤。
    # 真实极端行情的机制是"均值近似随机 + 高波动 + 尾部冲击"，而非每天稳定下跌。
    mu_s = drift_sign * 0.05 * sigma_s
    rng = np.random.default_rng(seed)
    shocks = mu_s + sigma_s * rng.standard_normal((days, n_paths))
    nav = np.cumprod(1 + shocks, axis=0)
    mdd_paths = (nav / np.maximum.accumulate(nav, axis=0) - 1).min(axis=0)
    shrink_paths = nav[-1] - 1
    return {
        "最大回撤": round(float(np.percentile(mdd_paths, 10)) * 100, 2),
        "缩水幅度": round(float(np.percentile(shrink_paths, 10)) * 100, 2),
        "年化波动率": round(float(sigma_s * np.sqrt(config.TRADING_DAYS)) * 100, 2),
        "修复周期(交易日)": "不适用",
    }


def _monte_carlo_navs(returns, stress_mult: float = 2.5, days: int = 60,
                      n_paths: int = 50, drift_sign: float = -1.0,
                      seed: int = 2024) -> dict:
    """蒙特卡洛模拟生成多条完整 NAV 路径（用于回撤曲线叠加展示）。"""
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 20:
        return {}
    sigma = float(r.std())
    if sigma <= 0:
        sigma = 0.01
    sigma_s = sigma * stress_mult
    mu_s = drift_sign * 0.05 * sigma_s
    rng = np.random.default_rng(seed)
    shocks = mu_s + sigma_s * rng.standard_normal((days, n_paths))
    nav = np.cumprod(1 + shocks, axis=0)
    dd = nav / np.maximum.accumulate(nav, axis=0) - 1
    worst_idx = int(np.argmin(dd.min(axis=0)))
    median_idx = int(np.argsort(dd.min(axis=0))[round(n_paths * 0.5)])
    return {"悲观路径(10%分位)": pd.Series(nav[:, worst_idx], name="蒙特卡洛悲观"),
            "中位路径": pd.Series(nav[:, median_idx], name="蒙特卡洛中位")}


def stress_test_portfolio(prices_dict: dict, weights: dict) -> dict:
    """对当前组合（权重）进行历史情景 + 蒙特卡洛压力测试。"""
    full_ret = _full_returns(prices_dict)
    port_full = _portfolio_returns(full_ret, weights)  # 组合全历史收益（MC 兜底用）
    results = {}
    for name, sc in config.STRESS_SCENARIOS.items():
        ret_df = _scenario_returns(prices_dict, sc["start"], sc["end"])
        if ret_df.empty:
            mc = _monte_carlo_pressure(port_full, stress_mult=sc["stress_mult"],
                                       days=sc["days"], drift_sign=sc["drift_sign"],
                                       seed=_stable_seed(name))
            if mc:
                mc["说明"] = f"{sc['desc']}｜组合无该区间历史，蒙特卡洛模拟替代"
                results[name] = mc
            else:
                results[name] = {"最大回撤": None, "缩水幅度": None, "年化波动率": None,
                                 "修复周期(交易日)": None, "说明": "区间数据不足"}
            continue
        port_ret = _portfolio_returns(ret_df, weights)
        nav = _nav_from_ret(port_ret)
        stats = _walk_stats(nav)
        stats["说明"] = sc["desc"]
        results[name] = stats
    # 追加参数化蒙特卡洛 what-if 情景
    for name, mc in config.MC_SCENARIOS.items():
        r = _monte_carlo_pressure(port_full, stress_mult=mc["stress_mult"], days=mc["days"],
                                  drift_sign=mc["drift_sign"], seed=_stable_seed("MC_" + name))
        if r:
            r["说明"] = mc["desc"]
            results[name] = r
    return results


def stress_test_fund(prices: pd.Series) -> dict:
    """单基金历史情景 + 蒙特卡洛压力测试。"""
    s = pd.to_numeric(prices, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    full_ret = s.pct_change().dropna()
    results = {}
    for name, sc in config.STRESS_SCENARIOS.items():
        sub = s[(s.index >= pd.Timestamp(sc["start"])) & (s.index <= pd.Timestamp(sc["end"]))]
        if len(sub) < 20:
            mc = _monte_carlo_pressure(full_ret, stress_mult=sc["stress_mult"],
                                       days=sc["days"], drift_sign=sc["drift_sign"],
                                       seed=_stable_seed(name))
            if mc:
                mc["说明"] = f"{sc['desc']}｜该基金无该区间历史，蒙特卡洛模拟替代"
                results[name] = mc
            else:
                results[name] = {"最大回撤": None, "缩水幅度": None, "年化波动率": None,
                                 "修复周期(交易日)": None, "说明": "区间数据不足"}
            continue
        nav = sub / sub.iloc[0]
        stats = _walk_stats(nav)
        stats["说明"] = sc["desc"]
        results[name] = stats
    # 追加参数化蒙特卡洛 what-if 情景
    for name, mc in config.MC_SCENARIOS.items():
        r = _monte_carlo_pressure(full_ret, stress_mult=mc["stress_mult"], days=mc["days"],
                                  drift_sign=mc["drift_sign"], seed=_stable_seed("MC_" + name))
        if r:
            r["说明"] = mc["desc"]
            results[name] = r
    return results


def stress_test_fund_with_paths(prices: pd.Series) -> tuple:
    """单基金压力测试，返回 (stats_dict, navs_dict)。

    navs_dict: {情景名: pd.Series(归一化净值)}
    """
    s = pd.to_numeric(prices, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    full_ret = s.pct_change().dropna()
    results = {}
    navs = {}
    for name, sc in config.STRESS_SCENARIOS.items():
        sub = s[(s.index >= pd.Timestamp(sc["start"])) & (s.index <= pd.Timestamp(sc["end"]))]
        if len(sub) < 20:
            mc = _monte_carlo_pressure(full_ret, stress_mult=sc["stress_mult"],
                                       days=sc["days"], drift_sign=sc["drift_sign"],
                                       seed=_stable_seed(name))
            if mc:
                mc["说明"] = f"{sc['desc']}｜该基金无该区间历史，蒙特卡洛模拟替代"
                results[name] = mc
                mc_paths = _monte_carlo_navs(full_ret, stress_mult=sc["stress_mult"],
                                             days=sc["days"], drift_sign=sc["drift_sign"],
                                             seed=_stable_seed(name))
                if mc_paths:
                    navs[name] = mc_paths["悲观路径(10%分位)"]
            else:
                results[name] = {"最大回撤": None, "缩水幅度": None, "年化波动率": None,
                                 "修复周期(交易日)": None, "说明": "区间数据不足"}
            continue
        nav = sub / sub.iloc[0]
        stats = _walk_stats(nav)
        stats["说明"] = sc["desc"]
        results[name] = stats
        navs[name] = nav
    for name, mc in config.MC_SCENARIOS.items():
        r = _monte_carlo_pressure(full_ret, stress_mult=mc["stress_mult"], days=mc["days"],
                                  drift_sign=mc["drift_sign"], seed=_stable_seed("MC_" + name))
        if r:
            r["说明"] = mc["desc"]
            results[name] = r
        mc_paths = _monte_carlo_navs(full_ret, stress_mult=mc["stress_mult"], days=mc["days"],
                                     drift_sign=mc["drift_sign"], seed=_stable_seed("MC_" + name))
        if mc_paths:
            navs[name] = mc_paths["悲观路径(10%分位)"]
    return results, navs


def stress_test_portfolio_with_paths(prices_dict: dict, weights: dict) -> tuple:
    """组合压力测试，返回 (stats_dict, navs_dict)。

    navs_dict: {情景名: pd.Series(归一化净值)}
    """
    full_ret = _full_returns(prices_dict)
    port_full = _portfolio_returns(full_ret, weights)
    results = {}
    navs = {}
    for name, sc in config.STRESS_SCENARIOS.items():
        ret_df = _scenario_returns(prices_dict, sc["start"], sc["end"])
        if ret_df.empty:
            mc = _monte_carlo_pressure(port_full, stress_mult=sc["stress_mult"],
                                       days=sc["days"], drift_sign=sc["drift_sign"],
                                       seed=_stable_seed(name))
            if mc:
                mc["说明"] = f"{sc['desc']}｜组合无该区间历史，蒙特卡洛模拟替代"
                results[name] = mc
                mc_paths = _monte_carlo_navs(port_full, stress_mult=sc["stress_mult"],
                                             days=sc["days"], drift_sign=sc["drift_sign"],
                                             seed=_stable_seed(name))
                if mc_paths:
                    navs[name] = mc_paths["悲观路径(10%分位)"]
            else:
                results[name] = {"最大回撤": None, "缩水幅度": None, "年化波动率": None,
                                 "修复周期(交易日)": None, "说明": "区间数据不足"}
            continue
        port_ret = _portfolio_returns(ret_df, weights)
        nav = _nav_from_ret(port_ret)
        stats = _walk_stats(nav)
        stats["说明"] = sc["desc"]
        results[name] = stats
        navs[name] = nav
    for name, mc in config.MC_SCENARIOS.items():
        r = _monte_carlo_pressure(port_full, stress_mult=mc["stress_mult"], days=mc["days"],
                                  drift_sign=mc["drift_sign"], seed=_stable_seed("MC_" + name))
        if r:
            r["说明"] = mc["desc"]
            results[name] = r
        mc_paths = _monte_carlo_navs(port_full, stress_mult=mc["stress_mult"], days=mc["days"],
                                     drift_sign=mc["drift_sign"], seed=_stable_seed("MC_" + name))
        if mc_paths:
            navs[name] = mc_paths["悲观路径(10%分位)"]
    return results, navs


def stress_summary(results: dict) -> str:
    """压力测试结论（抗冲击韧性评级）。"""
    mdd_vals = [abs(v["最大回撤"]) for v in results.values()
                if isinstance(v.get("最大回撤"), (int, float))]
    if not mdd_vals:
        return "压力测试数据不足，无法评级。"
    worst = max(mdd_vals)
    avg = sum(mdd_vals) / len(mdd_vals)
    if worst < 10:
        rating = "抗冲击能力：强"
    elif worst < 20:
        rating = "抗冲击能力：中等"
    elif worst < 30:
        rating = "抗冲击能力：偏弱"
    else:
        rating = "抗冲击能力：弱"
    return (f"{rating}。各情景平均最大回撤 {avg:.1f}%、最差情景回撤 {worst:.1f}%。"
            f"极端行情下资产可能缩水约 {worst:.1f}%，请结合自身承受力评估。")