# -*- coding: utf-8 -*-
"""指标计算层纯数值内核单元测试（合成确定性数据，无网络依赖）。"""
import unittest

import numpy as np
import pandas as pd

from fundquant import indicators as ind


class TestMaxDrawdown(unittest.TestCase):
    def test_known_values(self):
        prices = pd.Series([100.0, 120.0, 90.0, 110.0])
        # 回撤序列 = [0, 0, -0.25, -0.0833]，最小 -0.25
        self.assertAlmostEqual(ind.max_drawdown(prices), -0.25, places=6)

    def test_empty(self):
        self.assertEqual(ind.max_drawdown(pd.Series(dtype=float)), 0.0)


class TestLongestDrawdownDays(unittest.TestCase):
    def test_two_days_underwater(self):
        prices = pd.Series([1.0, 2.0, 1.0, 1.5, 3.0])
        self.assertEqual(ind.longest_drawdown_days(prices), 2)

    def test_empty(self):
        self.assertEqual(ind.longest_drawdown_days(pd.Series(dtype=float)), 0)


class TestRsi(unittest.TestCase):
    def test_monotonic_up_is_neutral_50(self):
        # 全程上涨：loss 恒 0 -> rs 恒 NaN -> fillna(50)
        prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        r = ind.rsi(prices, 3)
        self.assertEqual(len(r), len(prices))
        self.assertTrue(np.allclose(r.values, 50.0))

    def test_bounded_0_100(self):
        rng = np.random.default_rng(0)
        prices = pd.Series(100 + rng.normal(0, 3, 60).cumsum())
        r = ind.rsi(prices, 14)
        self.assertEqual(len(r), len(prices))
        self.assertTrue(((r >= 0.0) & (r <= 100.0) | r.isna()).all())


class TestValuationPercentileFromPe(unittest.TestCase):
    def test_max_pe_is_full_percentile(self):
        pe_hist = pd.DataFrame({"pe": np.arange(40, dtype=float)})
        # 当前 PE 为最大值 39，(pe < 39).mean() = 39/40 = 0.975
        self.assertAlmostEqual(ind.valuation_percentile_from_pe(pe_hist), 0.975, places=6)

    def test_insufficient_returns_nan(self):
        pe_hist = pd.DataFrame({"pe": np.arange(20, dtype=float)})
        self.assertTrue(np.isnan(ind.valuation_percentile_from_pe(pe_hist)))

    def test_missing_column_returns_nan(self):
        self.assertTrue(np.isnan(ind.valuation_percentile_from_pe(pd.DataFrame({"x": [1.0]}))))


class TestCumulativeAndPeriodReturn(unittest.TestCase):
    def test_cumulative_return(self):
        prices = pd.Series([1.0, 1.5, 1.2])
        self.assertAlmostEqual(ind.cumulative_return(prices), 0.2, places=6)

    def test_period_return(self):
        prices = pd.Series([1.0, 1.1, 1.21, 1.331])
        self.assertAlmostEqual(ind.period_return(prices, 1), 0.1, places=6)


if __name__ == "__main__":
    unittest.main()