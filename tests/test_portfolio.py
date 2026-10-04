# -*- coding: utf-8 -*-
"""投资组合优化层纯数值内核单元测试（合成确定性数据，无网络依赖）。"""
import unittest

import numpy as np
import pandas as pd

import config
from fundquant import portfolio as pf


def _ret_matrix(n=120, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    a = 1 + rng.normal(0.0005, 0.01, n)
    b = 1 + rng.normal(-0.0003, 0.008, n)
    c = 1 + rng.normal(0.0001, 0.012, n)
    px = pd.DataFrame({"A": a.cumprod(), "B": b.cumprod(), "C": c.cumprod()}, index=idx)
    return px.pct_change().dropna()


class TestExpectedReturns(unittest.TestCase):
    def test_shrinkage_math(self):
        ret = _ret_matrix(60)
        mu = ret.mean().values * config.TRADING_DAYS
        mu_bar = np.mean(mu)
        expected = mu_bar + (1.0 - config.EXPECTED_RETURN_SHRINK) * (mu - mu_bar)
        out = pf._expected_returns(ret)
        np.testing.assert_allclose(out, expected, rtol=1e-12)

    def test_cross_sectional_mean_preserved(self):
        ret = _ret_matrix(60)
        out = pf._expected_returns(ret)
        mu = ret.mean().values * config.TRADING_DAYS
        self.assertAlmostEqual(float(np.mean(out)), float(np.mean(mu)), places=10)


class TestBuildReturnMatrix(unittest.TestCase):
    def test_returns_cov_mean_shapes(self):
        ret = _ret_matrix(120)
        prices_dict = {c: (1 + ret[c]).cumprod() for c in ret.columns}
        r, cov, mu = pf.build_return_matrix(prices_dict)
        self.assertEqual(r.shape[1], 3)
        self.assertEqual(cov.shape, (3, 3))
        self.assertEqual(mu.shape, (3,))
        np.testing.assert_allclose(mu, r.mean().values * config.TRADING_DAYS, rtol=1e-12)


class TestOptimizePortfolio(unittest.TestCase):
    def test_min_variance_weights(self):
        ret = _ret_matrix(300)
        res = pf.optimize_portfolio(ret, mode="min_variance", max_weight=0.4)
        w = res["raw_weights"]
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)
        # 0.4*3=1.2>=1，上限未被放宽，权重应全部 <=0.4
        self.assertTrue((w <= 0.4 + 1e-9).all())
        self.assertTrue((w >= -1e-9).all())

    def test_max_sharpe_weights(self):
        ret = _ret_matrix(300)
        res = pf.optimize_portfolio(ret, mode="max_sharpe", max_weight=0.5)
        w = res["raw_weights"]
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)
        self.assertTrue((w <= 0.5 + 1e-9).all())

    def test_infeasible_max_weight_relaxes_with_note(self):
        ret = _ret_matrix(300)
        # 0.2*3=0.6<1，应自动放宽到等权上限 1/3 并给出提示
        res = pf.optimize_portfolio(ret, mode="min_variance", max_weight=0.2)
        w = res["raw_weights"]
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)
        self.assertTrue((w <= 1.0 / 3 + 1e-9).all())
        self.assertIsNotNone(res["note"])
        self.assertIn("放宽", res["note"])


class TestDrawdownAttribution(unittest.TestCase):
    def test_contributions_sum_to_mdd(self):
        idx = pd.date_range("2020-01-01", periods=50, freq="B")
        a = pd.Series(np.concatenate([np.linspace(1.0, 2.0, 25), np.linspace(2.0, 1.2, 25)]), index=idx)
        b = pd.Series(np.concatenate([np.linspace(1.0, 1.5, 25), np.linspace(1.5, 0.9, 25)]), index=idx)
        prices = {"A": a, "B": b}
        weights = {"A": 0.6, "B": 0.4}
        res = pf.drawdown_attribution(weights, prices)
        self.assertIn("mdd", res)
        self.assertEqual(set(res["contributions"].keys()), {"A", "B"})
        # 恒等式：Σ(成分贡献) ≈ 组合最大回撤（均在百分比口径）
        self.assertAlmostEqual(sum(res["contributions"].values()), res["mdd"], delta=0.1)

    def test_empty_input(self):
        self.assertEqual(pf.drawdown_attribution({}, {}), {})


if __name__ == "__main__":
    unittest.main()