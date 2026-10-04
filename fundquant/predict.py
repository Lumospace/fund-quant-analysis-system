# -*- coding: utf-8 -*-
"""AI 智能走势预测模型层（核心引擎之一）。

基于 LightGBM 的时序概率预测：输入历史行情 + 量化因子 + 波动特征，
采用时间序列切分（时序升序切 train/val/test）避免未来信息泄漏，
输出未来 N 日涨跌概率而非绝对涨跌，并附特征重要性、置信度、局限性说明。
"""
import numpy as np
import pandas as pd

from . import indicators as ind
from . import factors as fac

import config


def _build_features(prices: pd.Series, hist: pd.DataFrame,
                    pe_hist: pd.DataFrame = None,
                    ff: pd.DataFrame = None) -> pd.DataFrame:
    """构造特征矩阵（对齐价格索引）；pe_hist 供真实估值分位特征使用。

    ff 传入时复用已计算的因子矩阵，避免重复滚动计算。
    """
    if ff is None:
        ff = fac.compute_factor_frame(prices, hist, pe_hist)
    r = prices.pct_change()
    feats = pd.DataFrame(index=prices.index)
    # 技术特征
    feats["ret_5"] = prices.pct_change(5)
    feats["ret_20"] = prices.pct_change(20)
    feats["ret_60"] = prices.pct_change(60)
    feats["vol_20"] = r.rolling(20).std() * np.sqrt(252)
    feats["rsi_14"] = ind.rsi(prices, 14)
    feats["dist_ma20"] = prices / prices.rolling(20).mean() - 1
    feats["dist_ma60"] = prices / prices.rolling(60).mean() - 1
    feats["drawdown"] = prices / prices.cummax() - 1
    # 因子特征（valuation 因子 = -估值分位，故 val_pct 由其反推，避免重复滚动分位计算）
    for c in ff.columns:
        feats[f"f_{c}"] = ff[c]
    if "valuation" in ff.columns:
        feats["val_pct"] = -ff["valuation"]
    else:
        # 向量化窗口分位替代 rolling.apply(lambda)，口径同上（严格小于）
        feats["val_pct"] = prices.rolling(252).rank(pct=True) - 1.0 / 252
    # 去除全空列
    feats = feats.dropna(axis=1, how="all")
    return feats


def _make_dataset(X_full: pd.DataFrame, prices: pd.Series, horizon: int):
    """基于已构建的特征矩阵生成训练数据。

    特征工程只依赖 prices/hist、与 horizon 无关，故 X_full 一次构建后，
    多周期预测只需重复构造标签（未来 horizon 日收益是否为正）。
    """
    fwd = prices.shift(-horizon) / prices - 1
    # 未来 horizon 日窗口不足的尾部样本，标签未知，需标为 NaN（而非被 NaN>0 误判为 0），
    # 供下游训练/评估时剔除；其最新特征行仍需保留，用于最终对未来一点的预测。
    y = (fwd > 0).astype(float).where(fwd.notna())
    data = pd.concat([X_full, y.rename("label")], axis=1)
    # 只剔除特征暖机期的 NaN 行，保留尾部标签为 NaN 的最新样本。
    data = data[data.drop(columns=["label"]).notna().all(axis=1)]
    if len(data) < 120:
        return None, None
    X = data.drop(columns=["label"]).astype(float)
    return X, data["label"]


def _fit_model(X_train: pd.DataFrame, y_train: pd.Series):
    """训练 LightGBM 分类器，按正负样本比例温和加权处理类别不平衡。

    超参数按样本量自适应：n<200 从严防过拟合，n>500 充分学习。
    """
    import lightgbm as lgb
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = float(np.clip(neg / max(1, pos), 0.5, 2.0))
    n = X_train.shape[0]
    if n < 200:
        n_est, lr, depth, leaves, min_child = 100, 0.03, 3, 8, 20
    elif n < 500:
        n_est, lr, depth, leaves, min_child = 150, 0.04, 3, 12, 15
    else:
        n_est, lr, depth, leaves, min_child = 200, 0.05, 4, 20, 10
    model = lgb.LGBMClassifier(
        n_estimators=n_est, learning_rate=lr, max_depth=depth,
        num_leaves=leaves, min_child_samples=min_child, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbose=-1,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)
    return model


def _walk_forward_eval(X: pd.DataFrame, y: pd.Series, horizon: int,
                       n_splits: int = 5, embargo: int = 0):
    """expanding-window walk-forward 样本外校验（含时序 purging）。

    始终用「已可见」历史窗口训练、向后滚动预测，收集每个测试样本的
    样本外(OOS)概率，替代固定 60/80 单次切分带来的近端偏差。

    时序 purging：多日标签 y_t = 1{price[t+horizon] > price[t]}，训练样本 t
    的标签已用到 t~t+horizon 的未来信息。滚动外推时若训练样本的标签窗口与
    测试区间重叠，会造成时序泄漏。这里剔除标签窗口与测试区间重叠的训练样本
    （purge），可再按 embargo 额外留出间隔；purge 后训练样本不足时回退为不
    purge，保证短历史基金可用。
    返回 (OOS 概率数组[未覆盖处为 NaN], 初始训练窗口样本数, 最后一个可见窗口终点)。
    """
    n = len(X)
    min_train = max(120, int(n * 0.5))
    if n - min_train < 2 * n_splits:
        n_splits = 3
    probs = np.full(n, np.nan)
    bounds = np.unique(np.linspace(min_train, n, n_splits + 1).astype(int))
    min_effective = 120  # purge 后最少训练样本数，低于此值则放弃 purge
    for i in range(len(bounds) - 1):
        train_end, test_end = int(bounds[i]), int(bounds[i + 1])
        if test_end - train_end < 1:
            continue
        purge_end = train_end - horizon - embargo
        tr_slice = slice(0, purge_end) if purge_end >= min_effective else slice(0, train_end)
        # 训练只用标签已知的样本（剔除尾部未来窗口不足的 NaN 标签），避免喂给模型
        y_tr = y.iloc[tr_slice]
        valid_tr = y_tr.notna().to_numpy()
        model = _fit_model(X.iloc[tr_slice][valid_tr], y_tr[valid_tr])
        # 测试区间仅对标签已知的行填 OOS 概率，标签 NaN 行保持 NaN（下游掩码自动剔除）
        seg_pos = np.arange(train_end, test_end)
        valid_seg = y.iloc[seg_pos].notna().to_numpy()
        if valid_seg.any():
            probs[seg_pos[valid_seg]] = model.predict_proba(X.iloc[seg_pos[valid_seg]])[:, 1]
    # 返回最后一个可见训练窗口的终点，供最终预测模型复用（避免全样本重训泄漏未来信息）
    return probs, min_train, int(bounds[-2])


def _calibrate_probability(p_cal, t_cal, p_target) -> float:
    """用「样本外概率 → 真实标签」拟合 Isotonic 单调映射，校准树模型的极端概率。

    p_cal/t_cal 为 walk-forward 样本外(概率, 标签)，p_target 为待校准的最新原始概率。
    树模型 predict_proba 天然偏 0/1（叶子纯净），Isotonic 保序回归将其映射到真实频率；
    样本不足或标签单一（无法拟合单调关系）时回退原始概率，避免过度处理。
    """
    from sklearn.isotonic import IsotonicRegression
    p_cal = np.asarray(p_cal, dtype=float)
    t_cal = np.asarray(t_cal, dtype=float)
    if p_cal.shape[0] < 10 or len(np.unique(t_cal)) < 2:
        return float(p_target)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, t_cal)
    return float(np.clip(float(iso.predict([p_target])[0]), 0.05, 0.95))


def _train_predict(X: pd.DataFrame, y: pd.Series, prices: pd.Series, horizon: int) -> dict:
    """训练 LightGBM 并返回预测结果（X、y 已由 _make_dataset 构建）。

    稳健性改进：
    1. 用 walk-forward 滚动校验估计样本外 AUC/Brier，替代固定 60/80 单次切分；
    2. 按正负样本比例温和加权，缓解类别不平衡对概率的偏置；
    3. 最终模型复用 walk-forward 最后一个「已可见」训练窗口，避免全样本重训引入未来信息泄漏；
    4. 概率经 Isotonic 校准：用样本外(概率,标签)拟合单调映射，抑制树模型 0/1 极端输出，
       把上涨概率映射到更接近真实频率的区间（留出最后一个目标点，避免标签泄漏）。
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss

    try:
        oos_probs, min_train, last_train_end = _walk_forward_eval(X, y, horizon)
        mask = ~np.isnan(oos_probs)
        n_test = int(mask.sum())

        # 样本外稳健 AUC/Brier：默认随机水平，样本足够时才更新
        auc_oos, brier_oos = 0.5, 0.25
        if n_test > 2:
            if len(set(y[mask])) > 1:
                auc_oos = float(roc_auc_score(y[mask], oos_probs[mask]))
            brier_oos = float(brier_score_loss(y[mask], oos_probs[mask]))

        # Brier Skill Score：相对常数基线(基准涨率)的校准提升，1=完美、0=无改进、<0=差于基线
        base_rate = float(y[mask].mean()) if n_test > 2 else np.nan
        brier_skill = float(1.0 - brier_oos / (base_rate * (1 - base_rate))) \
            if n_test > 2 and 0 < base_rate < 1 else 0.0

        # 最终模型：复用 walk-forward 最后一个可见窗口（截至 last_train_end），预测最新一行
        y_last = y.iloc[:last_train_end]
        v_last = y_last.notna().to_numpy()
        model = _fit_model(X.iloc[:last_train_end][v_last], y_last[v_last])
        raw_prob = float(model.predict_proba(X.iloc[-1:])[0][1])
        imp = dict(zip(X.columns, model.feature_importances_))
        imp = dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:8])

        # 概率校准：留出最后一个目标点（raw_prob == oos_probs[-1]），用其余 OOS 点拟合 Isotonic
        p_vals = oos_probs[mask].astype(float)
        t_vals = y[mask].astype(float).to_numpy()
        prob_up = (_calibrate_probability(p_vals[:-1], t_vals[:-1], raw_prob)
                   if n_test >= 11 else raw_prob)

        # 置信度 = 样本外区分度(|AUC-0.5|，含反向) + 样本外概率校准(Brier) 加权合成
        brier_rel = max(0.0, 1.0 - brier_oos / 0.25)             # Brier=0→1，0.25(随机)→0
        discrimination = min(1.0, 2.0 * abs(auc_oos - 0.5))       # 反向信号同样计入
        confidence = 0.6 * discrimination + 0.4 * brier_rel
        confidence = round(min(0.95, confidence), 3)

        # 预期收益区间（用历史 horizon 收益分布）
        ret_dist = prices.pct_change(horizon).dropna()
        exp_ret = float(ret_dist.mean() * 100)
        lo = float(np.percentile(ret_dist, 10) * 100)
        hi = float(np.percentile(ret_dist, 90) * 100)

        return {
            "horizon": horizon, "prob_up": round(prob_up, 4),
            "prob_down": round(1 - prob_up, 4),
            "expected_return": round(exp_ret, 2),
            "return_range": (round(lo, 2), round(hi, 2)),
            "confidence": confidence, "auc": round(auc_oos, 3),
            "brier_skill": round(brier_skill, 3),
            "importance": imp, "n_train": min_train, "n_test": n_test,
            "method": "LightGBM walk-forward 时序分类",
        }
    except Exception as e:
        return _fallback(prices, horizon, f"模型训练失败:{str(e)[:40]}")


def forecast(prices: pd.Series, hist: pd.DataFrame, horizon: int = 20,
             verbose: bool = False, pe_hist: pd.DataFrame = None,
             ff: pd.DataFrame = None) -> dict:
    """预测未来 horizon 日涨跌概率（单周期入口）。

    返回: prob_up, prob_down, expected_return, confidence, auc, importance, n_train, n_test, method
    """
    X_full = _build_features(prices, hist, pe_hist, ff)
    X, y = _make_dataset(X_full, prices, horizon)
    if X is None:
        return _fallback(prices, horizon, "样本不足")
    return _train_predict(X, y, prices, horizon)


def _fallback(prices: pd.Series, horizon: int, reason: str) -> dict:
    """样本/训练失败时的朴素基线：近期动量 + 历史胜率。"""
    ret = prices.pct_change(horizon)
    hist_win = float((ret > 0).mean()) if len(ret) > 0 else 0.5
    mom = prices.pct_change(min(20, max(1, len(prices) - 1))).iloc[-1] if len(prices) > 1 else 0
    prob_up = 0.5 * hist_win + 0.5 * (0.5 + 0.3 * np.sign(mom))
    prob_up = float(np.clip(prob_up, 0.05, 0.95))
    ret_dist = ret.dropna()
    lo = float(np.percentile(ret_dist, 10) * 100) if len(ret_dist) > 10 else -5.0
    hi = float(np.percentile(ret_dist, 90) * 100) if len(ret_dist) > 10 else 5.0
    return {
        "horizon": horizon, "prob_up": round(prob_up, 4),
        "prob_down": round(1 - prob_up, 4),
        "expected_return": round(float(ret_dist.mean() * 100), 2) if len(ret_dist) > 0 else 0.0,
        "return_range": (round(lo, 2), round(hi, 2)),
        "confidence": 0.3, "auc": 0.5, "importance": {},
        "n_train": 0, "n_test": 0, "method": f"朴素基线（{reason}）",
    }


def multi_horizon_forecast(prices: pd.Series, hist: pd.DataFrame,
                           horizons=(7, 20, 60, 120),
                           pe_hist: pd.DataFrame = None,
                           ff: pd.DataFrame = None) -> dict:
    """多周期概率预测。特征矩阵只构建一次，各周期复用（仅标签不同）。"""
    X_full = _build_features(prices, hist, pe_hist, ff)
    out = {}
    for h in horizons:
        if len(prices) < h * 3:
            continue
        X, y = _make_dataset(X_full, prices, h)
        out[h] = _fallback(prices, h, "样本不足") if X is None else _train_predict(X, y, prices, h)
    return out


def trend_signals(prices: pd.Series) -> dict:
    """行情拐点/超买超卖信号。"""
    prices = pd.to_numeric(prices, errors="coerce").dropna()
    empty = {"rsi": None, "dist_ma20_pct": None, "signal": "数据不足",
             "support": None, "resistance": None}
    if len(prices) < 30:
        return empty
    rsi_now = float(ind.rsi(prices, 14).iloc[-1])
    ma20 = prices.rolling(20).mean()
    dist = float((prices.iloc[-1] / ma20.iloc[-1] - 1) * 100)
    if rsi_now < 30:
        signal = "超卖（可能反弹）"
    elif rsi_now > 70:
        signal = "超买（警惕回调）"
    elif dist > 5:
        signal = "偏离均线过高（追高风险）"
    elif dist < -5:
        signal = "跌破均线较深（超跌）"
    else:
        signal = "中性震荡"
    w = prices.tail(120)
    support = float(w.min())
    resist = float(w.max())
    return {"rsi": round(rsi_now, 1), "dist_ma20_pct": round(dist, 2),
            "signal": signal, "support": round(support, 4),
            "resistance": round(resist, 4)}


def calibration_curve_data(prices: pd.Series, hist: pd.DataFrame,
                           pe_hist: pd.DataFrame = None,
                           horizon: int = 20,
                           n_bins: int = 10,
                           ff: pd.DataFrame = None) -> dict:
    """概率校准曲线数据（Reliability Diagram）。

    使用 walk-forward 样本外概率，按预测概率分桶，计算每桶内实际正样本占比。
    返回 {bin_label: {pred_min, pred_max, pred_mean, actual_freq, count}}
    """
    X_full = _build_features(prices, hist, pe_hist, ff)
    X, y = _make_dataset(X_full, prices, horizon)
    if X is None or len(X) < 120:
        return {}
    oos_probs, _, _ = _walk_forward_eval(X, y, horizon)
    mask = ~np.isnan(oos_probs)
    if mask.sum() < 10:
        return {}
    preds = oos_probs[mask]
    actuals = y[mask].values.astype(int)
    bins = np.linspace(0, 1, n_bins + 1)
    result = {}
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        idx = (preds >= lo) & (preds < hi) if i < n_bins - 1 else (preds >= lo)
        cnt = int(idx.sum())
        if cnt < 2:
            continue
        pred_mean = float(preds[idx].mean())
        actual_freq = float(actuals[idx].mean())
        result[f"{lo*100:.0f}-{hi*100:.0f}%"] = {
            "pred_min": lo, "pred_max": hi,
            "pred_mean": round(pred_mean, 3),
            "actual_freq": round(actual_freq, 3),
            "count": cnt,
        }
    return result


def overfit_report(prices: pd.Series, hist: pd.DataFrame, horizon: int = 20,
                   pe_hist: pd.DataFrame = None, ff: pd.DataFrame = None,
                   n_perm: int = None) -> dict:
    """过拟合与稳健性诊断（样本内外对比 + 子样本稳健性 + 置换检验）。

    1. 样本内 vs 样本外 AUC 差距：样本内 AUC 远高 → 过拟合；
    2. 子样本稳健性：OOS 按时序拆前后两半分别算 AUC，波动大 → 结论不稳定；
    3. 置换检验：打乱训练标签重训 n_perm 次，真实 OOS AUC 显著高于随机基线
       （p 值小）说明捕捉到真实信号而非噪声。

    复用 walk-forward 的最后一个可见窗口做 holdout，训练窗做 purging
    （剔除标签窗口与 holdout 重叠的样本），避免置换/样本内口径的时序泄漏。
    返回诊断指标字典，样本不足时返回 {}。
    """
    from sklearn.metrics import roc_auc_score

    n_perm = n_perm if n_perm is not None else config.PERMUTATION_TEST_ITERS
    X_full = _build_features(prices, hist, pe_hist, ff)
    X, y = _make_dataset(X_full, prices, horizon)
    if X is None:
        return {}
    oos_probs, _, last_train_end = _walk_forward_eval(X, y, horizon)
    mask = ~np.isnan(oos_probs)
    if int(mask.sum()) < 10:
        return {}
    y_oos = y[mask].to_numpy().astype(int)
    p_oos = oos_probs[mask].astype(float)
    oos_auc = float(roc_auc_score(y_oos, p_oos)) if len(set(y_oos)) > 1 else 0.5

    # 训练窗（含 purge，避免训练标签窗口与 holdout 重叠）
    purge_end = last_train_end - horizon
    if purge_end >= 120:
        X_tr, y_tr = X.iloc[:purge_end], y.iloc[:purge_end]
    else:
        X_tr, y_tr = X.iloc[:last_train_end], y.iloc[:last_train_end]
    X_te, y_te = X.iloc[last_train_end:], y.iloc[last_train_end:]
    # 剔除尾部的标签为 NaN 的未来未知样本，避免 NaN 进入 astype(int) 与 AUC 计算
    v_te = y_te.notna().to_numpy()
    X_te, y_te = X_te[v_te], y_te[v_te]

    # 样本内 AUC（最终模型在其训练窗内的拟合优度）
    model = _fit_model(X_tr, y_tr)
    p_tr = model.predict_proba(X_tr)[:, 1]
    is_auc = float(roc_auc_score(y_tr.to_numpy(), p_tr)) if len(set(y_tr.to_numpy())) > 1 else 0.5

    # 子样本稳健性：OOS 按时序拆前后两半
    half = len(y_oos) // 2

    def _auc(a, b):
        return float(roc_auc_score(a, b)) if len(set(a)) > 1 else np.nan

    sub1 = _auc(y_oos[:half], p_oos[:half]) if half > 1 else np.nan
    sub2 = _auc(y_oos[half:], p_oos[half:]) if len(y_oos) > half + 1 else np.nan

    # 置换检验：测试集不变，打乱训练标签重训，构造「无真实信号」的 AUC 零分布
    y_te_arr = y_te.to_numpy().astype(int)
    holdout_auc, perm_baseline, p_value = np.nan, None, np.nan
    if len(y_te_arr) >= 10 and len(set(y_te_arr)) > 1:
        holdout_auc = float(roc_auc_score(y_te_arr, model.predict_proba(X_te)[:, 1]))
        rng = np.random.default_rng(42)
        perm_aucs = []
        for _ in range(int(n_perm)):
            y_shuf = pd.Series(rng.permutation(y_tr.to_numpy()), index=y_tr.index)
            m = _fit_model(X_tr, y_shuf)
            perm_aucs.append(float(roc_auc_score(y_te_arr, m.predict_proba(X_te)[:, 1])))
        perm_aucs = np.asarray(perm_aucs, dtype=float)
        perm_baseline = float(perm_aucs.mean())
        p_value = float((perm_aucs >= holdout_auc).mean())

    return {
        "oos_auc": round(oos_auc, 3),
        "is_auc": round(is_auc, 3),
        "gap": round(is_auc - oos_auc, 3),
        "sub_first": None if np.isnan(sub1) else round(sub1, 3),
        "sub_second": None if np.isnan(sub2) else round(sub2, 3),
        "holdout_auc": None if np.isnan(holdout_auc) else round(holdout_auc, 3),
        "perm_baseline": None if perm_baseline is None else round(perm_baseline, 3),
        "perm_p": None if np.isnan(p_value) else round(p_value, 3),
        "n_perm": int(n_perm), "n_train": int(len(y_tr)), "n_holdout": int(len(y_te_arr)),
    }