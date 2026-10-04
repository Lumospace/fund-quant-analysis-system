# -*- coding: utf-8 -*-
"""因子工程层纯数值内核单元测试（合成确定性数据，无网络依赖）。"""
import unittest

import numpy as np
import pandas as pd

from fundquant import factors as fac


def _price_index(n=260):
    return pd.date_range("2020-01-01", periods=n, freq="B")


def _factor_frame(prices, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {k: rng.normal(size=len(prices)) for k in fac.FACTORS.keys()},
        index=prices.index,
    )


class TestFactorIc(unittest.TestCase):
    def test_perfectly_monotonic_is_one(self):
        factor = pd.Series(np.arange(1.0, 51.0))
        fwd = pd.Series(np.arange(1.0, 51.0))
        self.assertAlmostEqual(fac.factor_ic(factor, fwd), 1.0, places=6)

    def test_insufficient_returns_nan(self):
        factor = pd.Series(np.arange(10.0))
        fwd = pd.Series(np.arange(10.0))
        self.assertTrue(np.isnan(fac.factor_ic(factor, fwd)))


class TestFactorIr(unittest.TestCase):
    def test_known_value(self):
        # mean=0.2, std(ddof=1)=0.1 -> IR=2.0
        ic = pd.Series([0.1, 0.2, 0.3])
        self.assertAlmostEqual(fac.factor_ir(ic), 2.0, places=6)

    def test_short_series_is_zero(self):
        self.assertEqual(fac.factor_ir(pd.Series([0.1])), 0.0)

    def test_constant_series_is_zero(self):
        self.assertEqual(fac.factor_ir(pd.Series([0.1, 0.1, 0.1])), 0.0)


class TestFactorMonotonicity(unittest.TestCase):
    def test_insufficient_returns_nan(self):
        # m.sum() == 10 < n_group*8(=40)
        factor = pd.Series(np.arange(10.0))
        fwd = pd.Series(np.arange(10.0))
        self.assertTrue(np.isnan(fac.factor_monotonicity(factor, fwd)))

    def test_perfectly_monotonic_is_one(self):
        factor = pd.Series(np.arange(100.0))
        fwd = pd.Series(np.arange(100.0))
        self.assertAlmostEqual(fac.factor_monotonicity(factor, fwd, n_group=5), 1.0, places=6)


class TestValidateFactors(unittest.TestCase):
    def test_structure_and_status(self):
        prices = pd.Series(np.linspace(100.0, 150.0, 260), index=_price_index())
        ff = _factor_frame(prices)
        df = fac.validate_factors(prices, None, ff=ff)
        cols = list(df.columns)
        for c in ["因子", "类别", "方向", "IC", "IR", "t值", "多头超额胜率",
                  "单调性", "连续失效月数", "状态", "说明"]:
            self.assertIn(c, cols)
        self.assertEqual(len(df), len(fac.FACTORS))
        self.assertTrue(df["状态"].isin({"样本不足", "已下线", "预警", "有效"}).all())
        self.assertTrue(df["方向"].isin({"正向", "负向"}).all())


class TestFactorDecay(unittest.TestCase):
    def test_index_and_columns(self):
        prices = pd.Series(np.linspace(100.0, 150.0, 260), index=_price_index())
        ff = _factor_frame(prices)
        horizons = (5, 10, 20, 60, 120)
        df = fac.factor_decay(prices, None, horizons=horizons, ff=ff)
        self.assertEqual(list(df.index), list(horizons))
        expected = {f.name for f in fac.FACTORS.values()}
        self.assertEqual(set(df.columns), expected)


class TestFactorQuantileReturns(unittest.TestCase):
    def test_each_factor_has_all_groups(self):
        prices = pd.Series(np.linspace(100.0, 150.0, 260), index=_price_index())
        ff = _factor_frame(prices)
        out = fac.factor_quantile_returns(prices, None, forward=20, n_group=5, ff=ff)
        self.assertEqual(set(out.keys()), {f.name for f in fac.FACTORS.values()})
        for name, groups in out.items():
            self.assertIsNotNone(groups, name)
            self.assertEqual(set(groups.keys()), {"Q1", "Q2", "Q3", "Q4", "Q5"})


class TestFactorDriftEvents(unittest.TestCase):
    def test_events_detected(self):
        history = [
            {"date": "2024-01-01", "factors": {"动量因子": "有效", "波动率因子": "有效"}},
            {"date": "2024-01-02", "factors": {"动量因子": "预警", "波动率因子": "有效"}},
            {"date": "2024-01-03", "factors": {"动量因子": "预警", "波动率因子": "已下线"}},
        ]
        tbl, events = fac.factor_drift_events(history)
        self.assertEqual(list(tbl.columns), ["日期", "动量因子", "波动率因子"])
        self.assertEqual(len(tbl), 3)
        self.assertEqual(events, [
            ("2024-01-02", "动量因子", "有效", "预警"),
            ("2024-01-03", "波动率因子", "有效", "已下线"),
        ])

    def test_empty_history(self):
        tbl, events = fac.factor_drift_events([])
        self.assertIsNone(tbl)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()