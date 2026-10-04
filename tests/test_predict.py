# -*- coding: utf-8 -*-
"""预测模型层数据构造 / walk-forward 时序切分单元测试（合成确定性数据，无网络依赖）。"""
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from fundquant import predict


class TestMakeDataset(unittest.TestCase):
    def _mk(self, n, start_price, end_price, horizon=20):
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        prices = pd.Series(np.linspace(start_price, end_price, n), index=idx)
        X_full = pd.DataFrame({"f1": np.arange(n, dtype=float)}, index=idx)
        fwd = prices.shift(-horizon) / prices - 1
        return X_full, prices, fwd

    def test_labels_match_sign_and_tail_is_nan(self):
        X_full, prices, fwd = self._mk(200, 100.0, 300.0, 20)
        X, y = predict._make_dataset(X_full, prices, horizon=20)
        valid = fwd.notna().to_numpy()
        expected = (fwd.to_numpy()[valid] > 0).astype(float)
        # 有效区标签 = 未来收益符号（上涨 → 1）
        self.assertTrue(np.array_equal(y.to_numpy()[valid], expected))
        # 尾部（未来窗口不足）必须为 NaN，而非被误标为 0
        self.assertTrue(np.isnan(y.to_numpy()[~valid]).all())
        self.assertEqual(X.shape[1], 1)

    def test_decreasing_prices_all_zero(self):
        X_full, prices, fwd = self._mk(200, 300.0, 100.0, 20)
        X, y = predict._make_dataset(X_full, prices, horizon=20)
        valid = fwd.notna().to_numpy()
        self.assertTrue((y.to_numpy()[valid] == 0.0).all())

    def test_insufficient_returns_none(self):
        X_full, prices, _ = self._mk(100, 100.0, 200.0, 20)
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

    def test_tail_nan_labels_excluded(self):
        n = 400
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        X = pd.DataFrame({"f1": np.arange(n, dtype=float)}, index=idx)
        y = pd.Series((np.arange(n) % 2).astype(float), index=idx)
        y.iloc[-20:] = np.nan  # 模拟未来窗口不足的尾部标签

        def fake_fit(Xt, yt):
            # 训练绝不收到 NaN 标签
            assert not np.isnan(yt.to_numpy()).any(), "训练集混入了 NaN 标签"
            class _M:
                def predict_proba(self, Z):
                    return np.column_stack([0.5 * np.ones(len(Z)), 0.5 * np.ones(len(Z))])
            return _M()

        with mock.patch.object(predict, "_fit_model", fake_fit):
            probs, _, _ = predict._walk_forward_eval(X, y, horizon=20)

        # 标签未知的尾部 20 行不应得到 OOS 概率
        self.assertTrue(np.isnan(probs[-20:]).all())
        self.assertTrue(np.isnan(probs[:-20]).sum() == 200)


if __name__ == "__main__":
    unittest.main()