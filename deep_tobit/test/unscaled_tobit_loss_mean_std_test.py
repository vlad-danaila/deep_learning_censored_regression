import unittest
import portion as p
import numpy as np
import torch as t
from deep_tobit.util import normalize, unnormalize, to_torch
from deep_tobit.loss import Unscaled_Tobit_Loss
import math

ENABLE_LONG_RUNNING_TESTS = True

class TobitOptimizationTest(unittest.TestCase):

    def read_tensors_from_intervals(self, y_intervals):
        single_valued_list, left_censored_list, right_censored_list = [], [], []

        for interval in y_intervals:
            # If single valued interval
            if interval.lower == interval.upper:
                single_valued_list.append(interval.lower)
            elif interval.upper == p.inf:
                right_censored_list.append(interval.lower)
            elif interval.lower == 0 or interval.lower == -p.inf:
                left_censored_list.append(interval.upper)
            else:
                raise Exception('Uncensored interval encountered', interval)

        all = np.array(single_valued_list + left_censored_list + right_censored_list)
        mean, std = all.mean(), all.std() + 1e-10

        y_single_valued = t.tensor(single_valued_list, dtype=t.float)
        y_left_censored = t.tensor(left_censored_list, dtype=t.float)
        y_right_censored = t.tensor(right_censored_list, dtype=t.float)

        y_single_valued = normalize(y_single_valued, mean, std)
        y_left_censored = normalize(y_left_censored, mean, std)
        y_right_censored = normalize(y_right_censored, mean, std)

        return y_single_valued, y_left_censored, y_right_censored, mean, std

    def fit_mean_std_with_tobit(self, y_intervals):
        y_single_valued, y_left_censored, y_right_censored, data_mean, data_std = self.read_tensors_from_intervals(
            y_intervals)
        delta = to_torch(0, grad=True)
        # tuple for single valued, left censored, right censored
        x_tuple = (delta, delta, delta)
        y_tuple = (y_single_valued, y_left_censored, y_right_censored)
        tobit = Unscaled_Tobit_Loss(device = 'cpu')
        optimizer = t.optim.SGD([delta], lr=1e-1)
        patience = 5
        for i in range(10_000):
            prev_delta = delta.clone()
            optimizer.zero_grad()
            loss = tobit(x_tuple, y_tuple)
            loss.backward()
            optimizer.step()
            early_stop = math.fabs(delta - prev_delta) < 1e-5
            if early_stop:
                patience -= 1
                if patience == 0:
                    break
            else:
                patience = 5
            if i % 100 == 0:
                print(i, delta)
        mean = unnormalize(delta, data_mean, data_std)
        return mean

    def check_mean_std(self, y_intervals, expected_mean, delta_real = .4):
        mean = self.fit_mean_std_with_tobit(y_intervals)
        print(f'mean = {mean}')
        self.assertAlmostEqual(mean.item(), expected_mean, delta = delta_real)

    # 1) 20 30 40
    def test_single_valued_only(self):
        y_intervals = [p.singleton(20), p.singleton(30), p.singleton(40)]
        self.check_mean_std(y_intervals, 30)

    # 2) 30 >50 >50
    def test_right_censored_30_50_50(self):
        y_intervals = [p.singleton(30), p.closed(50, p.inf), p.closed(50, p.inf)]
        self.check_mean_std(y_intervals, 47.86)

    # 3) 30 >50 >50 >50 >50
    def test_right_censored_30_50_50_50_50(self):
        y_intervals = [p.singleton(30), p.closed(50, p.inf), p.closed(50, p.inf), p.closed(50, p.inf), p.closed(50, p.inf)]
        self.check_mean_std(y_intervals, 51.6)

    # 4) 30 30 >50
    def test_right_censored_30_30_50(self):
        y_intervals = [p.singleton(30), p.singleton(30), p.closed(50, p.inf)]
        self.check_mean_std(y_intervals, 38.16)

    # 5) 30 30 30 30 >50
    def test_right_censored_30_30_30_30_50(self):
        y_intervals = [p.singleton(30), p.singleton(30), p.singleton(30), p.singleton(30), p.closed(50, p.inf)]
        self.check_mean_std(y_intervals, 34.8353)

    # 6) 50 >30 >30
    def test_right_censored_50_30_30(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.singleton(50), p.closed(30, p.inf), p.closed(30, p.inf)]
            self.check_mean_std(y_intervals, 50.68)

    # 7) 40 50 >30 >30
    def test_right_censored_40_50_30_30(self):
        y_intervals = [p.singleton(50), p.singleton(40), p.closed(30, p.inf), p.closed(30, p.inf)]
        self.check_mean_std(y_intervals, 45.58)

    # 8) >30 >30 >30
    def test_all_right_censored(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.closed(30, p.inf), p.closed(30, p.inf), p.closed(30, p.inf)]
            self.check_mean_std(y_intervals, 30)

    # 9) <10 <10 30
    def test_left_censored_10_10_30(self):
        y_intervals = [p.closed(-p.inf, 10), p.closed(-p.inf, 10), p.singleton(30)]
        self.check_mean_std(y_intervals, 12.13)

    # 10) <10 <10 <10 <10 30
    def test_left_censored_10_10_10_10_30(self):
        y_intervals = [ p.closed(-p.inf, 10), p.closed(-p.inf, 10), p.closed(-p.inf, 10), p.closed(-p.inf, 10), p.singleton(30)]
        self.check_mean_std(y_intervals, 8.39)

    # 11) <10 30 30
    def test_left_censored_10_30_30(self):
        y_intervals = [p.closed(-p.inf, 10), p.singleton(30), p.singleton(30)]
        self.check_mean_std(y_intervals, 21.83)

    # 12) <10 30 30 30 30
    def test_left_censored_10_30_30_30_30(self):
        y_intervals = [p.closed(-p.inf, 10), p.singleton(30), p.singleton(30), p.singleton(30), p.singleton(30)]
        self.check_mean_std(y_intervals, 25.1465)

    # 13) 10 <30 <30
    def test_left_censored_30_30_10(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.singleton(10)]
            self.check_mean_std(y_intervals, 9.31)

    # 14) 10 <30 <30 <30 <30
    def test_left_censored_30_30_30_30_10(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.singleton(10)]
            self.check_mean_std(y_intervals, 9.51)

    # 15) 10 20 <30
    def test_left_censored_30_20_10(self):
        y_intervals = [p.closed(-p.inf, 30), p.singleton(10), p.singleton(20)]
        self.check_mean_std(y_intervals, 14.9935)

    # 16) 10 20 <30 <30 <30 <30
    def test_left_censored_30_30_30_30_10_20(self):
        y_intervals = [p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.singleton(10), p.singleton(20)]
        self.check_mean_std(y_intervals, 14.25)

    # 17) <30 <30 <30
    def test_left_censored_30_30_30(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.closed(-p.inf, 30)]
            self.check_mean_std(y_intervals, 30.0000)

    # 18) <10 <20 30
    def test_left_censored_10_20_30(self):
        y_intervals = [p.closed(-p.inf, 10), p.closed(-p.inf, 20), p.singleton(30)]
        self.check_mean_std(y_intervals, 15.58)

    # 19) 30 >40 >50
    def test_right_censored_30_40_50(self):
        y_intervals = [p.singleton(30), p.closed(40, p.inf), p.closed(50, p.inf)]
        self.check_mean_std(y_intervals, 44.41)

    # 20) >30 40 >50
    def test_right_censored_40_30_50(self):
        y_intervals = [p.closed(30, p.inf), p.singleton(40), p.closed(50, p.inf)]
        self.check_mean_std(y_intervals, 48.4679)

    # 21) >30 40 40 40 >50
    def test_right_censored_30_40_40_40_50(self):
        y_intervals = [p.closed(30, p.inf), p.singleton(40), p.singleton(40), p.singleton(40), p.closed(50, p.inf)]
        self.check_mean_std(y_intervals, 43.1742)

    # 22) >30 40 <50
    def test_left_right_censored_30_40_50(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.closed(30, p.inf), p.singleton(40), p.closed(-p.inf, 50)]
            self.check_mean_std(y_intervals, 40)

    # 23) >20 >30 40 <50
    def test_left_right_censored_20_30_40_50(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.closed(20, p.inf), p.closed(30, p.inf), p.singleton(40), p.closed(-p.inf, 50)]
            self.check_mean_std(y_intervals, 40.47)

    # 24) >20 >30 <50
    def test_left_right_censored_20_30_50(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.closed(20, p.inf), p.closed(30, p.inf), p.closed(-p.inf, 50)]
            self.check_mean_std(y_intervals, 41.39)

    # 25) <20 >30
    def test_diverge_20_30(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.closed(-p.inf, 20), p.closed(30, p.inf)]
            # is the negative std correct ?
            self.check_mean_std(y_intervals, 25)

    # interesting case only the lower 25 bound is taken into account
    # 26) <20 >25 >50 >50 >50
    def test_diverge_20_25_50_50_50(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.closed(-p.inf, 20), p.closed(25, p.inf), p.closed(50, p.inf), p.closed(50, p.inf), p.closed(50, p.inf)]
            self.check_mean_std(y_intervals, 49.57)

    # 27) >30 35 37 <50
    def test_left_right_censoring_30_35_37_50(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.closed(30, p.inf), p.singleton(35), p.singleton(37), p.closed(-p.inf, 50)]
            self.check_mean_std(y_intervals, 36.84)

    # 28) <30 32 34 >50
    def test_divergent_30_32_34_50(self):
        y_intervals = [p.closed(-p.inf, 30), p.singleton(32), p.singleton(34), p.closed(50, p.inf)]
        self.check_mean_std(y_intervals, 36.16)

    # 29) <10 <30 40 42 44
    def test_left_censored_data(self):
        y_intervals = [p.closed(-p.inf, 10), p.closed(-p.inf, 30), p.singleton(40), p.singleton(42), p.singleton(44)]
        self.check_mean_std(y_intervals, 30.14)

    # 30) <10 <30 40 42 44 >60 >70
    def test_left_right_censoring_10_30_40_42_44_60_70(self):
        y_intervals = [p.closed(-p.inf, 10), p.closed(-p.inf, 30), p.singleton(40), p.singleton(42), p.singleton(44),
                p.closed(60, p.inf), p.closed(70, p.inf)]
        self.check_mean_std(y_intervals, 42.1983)