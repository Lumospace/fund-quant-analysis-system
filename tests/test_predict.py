# -*- coding: utf-8 -*-
"""预测模型层数据构造 / walk-forward 时序切分单元测试（合成确定性数据，无网络依赖）。"""
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from fundquant import predict


class TestMakeDataset(unittest.TestCase):
    def test_increasing_prices_labels_match_sign(self):
        idx = pd.date_range("2020-01-01", periods=200, freq="B")
        prices = pd.Series(np.linspace(100.0, 200.0, 200), index=idx)
        X_full = pd.DataFrame({"f1": np.arange(200, dtype=float)}, index=idx)
        X, y = predict._make_dataset(X_full, prices, horizon=20)
        # 有效区（未来窗口已知）标签必须严格等于未来收益符号
        fwd = prices.shift(-20) / prices - 1
        valid = fwd.notna()
        expected = (fwd[valid] > 0).astype(int)
        self.assertTrue((y.loc[valid].astype(int) == expected).all())
        self.assertEqual(X.shape[1], 1)

    def test_decreasing_prices_labels_match_sign(self):
        idx = pd.date_range("2020-01-01", periods=200, freq="B")
        prices = pd.Series(np.linspace(200.0, 100.0, 200), index=idx)
        X_full = pd.DataFrame({"f1": np.arange(200, dtype=float)}, index=idx)
        X, y = predict._make_dataset(X_full, prices, horizon=20)
        fwd = prices.shift(-20) / prices - 1
        valid = fwd.notna()
        expected = (fwd[valid] > 0).astype(int)
        self.assertTrue((y.loc[valid].astype(int) == expected).all())
        self.assertTrue((y.loc[valid] == 0).all())

    def test_insufficient_returns_none(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        prices = pd.Series(np.linspace(100.0, 110.0, 100), index=idx)
        X_full = pd.DataFrame({"f1": np.arange(100, dtype=float)}, index=idx)
        self.assertEqual(predict._make_dataset(X_full, prices, 20), (None, None))


class TestWalkForwardEval(unittest.TestCase):
    def test_purging_applied(self):
        n = 400
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        X = pd.DataFrame({"f1": np.arange(n, dtype=float)}, index=idx)
        y = pd.Series((np.arange(n) % 2).astype(int), index=idx)

        calls = []

        def fake_fit(Xt, yt):
            calls.append(int(len(Xt)))

            class _M:
                def predict_proba(self, Z):
                    return np.column_stack([0.5 * np.ones(len(Z)), 0.5 * np.ones(len(Z))])

            return _M()

        with mock.patch.object(predict, "_fit_model", fake_fit):
            probs, min_train, last_end = predict._walk_forward_eval(X, y, horizon=20)

        self.assertEqual(min_train, 200)
        self.assertEqual(last_end, 360)
        self.assertEqual(probs.shape, (n,))
        # 首个训练窗：train_end=200，purge 后应回退到 200-20=180 < 200，证明 purging 生效
        self.assertEqual(calls[0], 180)
        # 首 min_train 个样本未覆盖（NaN），其余被 OOS 概率覆盖
        self.assertTrue(np.isnan(probs[:200]).all())
        self.assertTrue(np.isnan(probs[200:]).sum() == 0)


if __name__ == "__main__":
    unittest.main()