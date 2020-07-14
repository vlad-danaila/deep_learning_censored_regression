import unittest
import portion as p
import numpy as np
import torch as t
from implementation.util import normalize, unnormalize, to_torch
from implementation.tobit_loss_module import TobitLoss
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
        tobit = TobitLoss(device='cpu')
        optimizer = t.optim.SGD([delta, tobit.gamma], lr=1e-1)
        patience = 5
        for i in range(10_000):
            prev_delta, prev_gamma = delta.clone(), tobit.gamma.clone()
            optimizer.zero_grad()
            loss = tobit(x_tuple, y_tuple)
            loss.backward()
            optimizer.step()
            early_stop = math.fabs(delta - prev_delta) + math.fabs(tobit.gamma - prev_gamma) < 1e-5
            if early_stop:
                patience -= 1
                if patience == 0:
                    break
            else:
                patience = 5
            if i % 100 == 0:
                print(i, delta, tobit.gamma)
        mean, std = delta / tobit.gamma, 1 / tobit.gamma
        mean, std = unnormalize(mean, data_mean, data_std), std * data_std
        return mean, std

    def check_mean_std(self, y_intervals, expected_mean, expected_std, delta_real = .4):
        mean, std = self.fit_mean_std_with_tobit(y_intervals)
        self.assertAlmostEqual(mean.item(), expected_mean, delta = delta_real)
        self.assertAlmostEqual(std.item(), expected_std, delta = delta_real)

    # 1) 20 30 40
    def test_single_valued_only(self):
        y_intervals = [p.singleton(20), p.singleton(30), p.singleton(40)]
        self.check_mean_std(y_intervals, 30, 8.1650)

    # 2) 30 >50 >50
    def test_right_censored_30_50_50(self):
        y_intervals = [p.singleton(30), p.closed(50, p.inf), p.closed(50, p.inf)]
        self.check_mean_std(y_intervals, 58.0191, 23.6987)

    # 3) 30 >50 >50 >50 >50
    def test_right_censored_30_50_50_50_50(self):
        y_intervals = [p.singleton(30), p.closed(50, p.inf), p.closed(50, p.inf), p.closed(50, p.inf), p.closed(50, p.inf)]
        self.check_mean_std(y_intervals, 73.3895, 29.4739)

    # 4) 30 30 >50
    def test_right_censored_30_30_50(self):
        y_intervals = [p.singleton(30), p.singleton(30), p.closed(50, p.inf)]
        self.check_mean_std(y_intervals, 39.2030, 13.5950)

    # 5) 30 30 30 30 >50
    def test_right_censored_30_30_30_30_50(self):
        y_intervals = [p.singleton(30), p.singleton(30), p.singleton(30), p.singleton(30), p.closed(50, p.inf)]
        self.check_mean_std(y_intervals, 34.8353, 9.8508)

    # 6) 50 >30 >30
    def test_right_censored_50_30_30(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.singleton(50), p.closed(30, p.inf), p.closed(30, p.inf)]
            self.check_mean_std(y_intervals, 49.9330, 0.3652)

    # 7) 40 50 >30 >30
    def test_right_censored_40_50_30_30(self):
        y_intervals = [p.singleton(50), p.singleton(40), p.closed(30, p.inf), p.closed(30, p.inf)]
        self.check_mean_std(y_intervals, 45.0124, 4.9814)

    # 8) >30 >30 >30
    def test_all_right_censored(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.closed(30, p.inf), p.closed(30, p.inf), p.closed(30, p.inf)]
            self.check_mean_std(y_intervals, 30, 1e-10)

    # 9) <10 <10 30
    def test_left_censored_10_10_30(self):
        y_intervals = [p.closed(-p.inf, 10), p.closed(-p.inf, 10), p.singleton(30)]
        self.check_mean_std(y_intervals, 1.8617, 23.7228)

    # 10) <10 <10 <10 <10 30
    def test_left_censored_10_10_10_10_30(self):
        y_intervals = [ p.closed(-p.inf, 10), p.closed(-p.inf, 10), p.closed(-p.inf, 10), p.closed(-p.inf, 10), p.singleton(30)]
        self.check_mean_std(y_intervals, -13.5183, 29.5021)

    # 11) <10 30 30
    def test_left_censored_10_30_30(self):
        y_intervals = [p.closed(-p.inf, 10), p.singleton(30), p.singleton(30)]
        self.check_mean_std(y_intervals, 20.7514, 13.6005)

    # 12) <10 30 30 30 30
    def test_left_censored_10_30_30_30_30(self):
        y_intervals = [p.closed(-p.inf, 10), p.singleton(30), p.singleton(30), p.singleton(30), p.singleton(30)]
        self.check_mean_std(y_intervals, 25.1465, 9.8524)

    # 13) 10 <30 <30
    def test_left_censored_30_30_10(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.singleton(10)]
            self.check_mean_std(y_intervals, 10.0067, 0.3653)

    # 14) 10 <30 <30 <30 <30
    def test_left_censored_30_30_30_30_10(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.singleton(10)]
            self.check_mean_std(y_intervals, 10.0055, 0.3325)

    # 15) 10 20 <30
    def test_left_censored_30_20_10(self):
        y_intervals = [p.closed(-p.inf, 30), p.singleton(10), p.singleton(20)]
        self.check_mean_std(y_intervals, 14.9935, 4.9903)

    # 16) 10 20 <30 <30 <30 <30
    def test_left_censored_30_30_30_30_10_20(self):
        y_intervals = [p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.singleton(10), p.singleton(20)]
        self.check_mean_std(y_intervals, 14.9774, 4.9659)

    # 17) <30 <30 <30
    def test_left_censored_30_30_30(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.closed(-p.inf, 30)]
            self.check_mean_std(y_intervals, 30.0000, 1.0000e-10)

    # 18) <10 <20 30
    def test_left_censored_10_20_30(self):
        y_intervals = [p.closed(-p.inf, 10), p.closed(-p.inf, 20), p.singleton(30)]
        self.check_mean_std(y_intervals, 7.9252, 18.9532)

    # 19) 30 >40 >50
    def test_right_censored_30_40_50(self):
        y_intervals = [p.singleton(30), p.closed(40, p.inf), p.closed(50, p.inf)]
        self.check_mean_std(y_intervals, 52.0748, 18.9532)

    # 20) >30 40 >50
    def test_right_censored_40_30_50(self):
        y_intervals = [p.closed(30, p.inf), p.singleton(40), p.closed(50, p.inf)]
        self.check_mean_std(y_intervals, 48.4679, 8.7341)

    # 21) >30 40 40 40 >50
    def test_right_censored_30_40_40_40_50(self):
        y_intervals = [p.closed(30, p.inf), p.singleton(40), p.singleton(40), p.singleton(40), p.closed(50, p.inf)]
        self.check_mean_std(y_intervals, 43.1742, 5.5445)

    # 22) >30 40 <50
    def test_left_right_censored_30_40_50(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.closed(30, p.inf), p.singleton(40), p.closed(-p.inf, 50)]
            self.check_mean_std(y_intervals, 40, 0.1714)

    # 23) >20 >30 40 <50
    def test_left_right_censored_20_30_40_50(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.closed(20, p.inf), p.closed(30, p.inf), p.singleton(40), p.closed(-p.inf, 50)]
            self.check_mean_std(y_intervals, 39.9970, 0.2393)

    # 24) >20 >30 <50
    def test_left_right_censored_20_30_50(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.closed(20, p.inf), p.closed(30, p.inf), p.closed(-p.inf, 50)]
            self.check_mean_std(y_intervals, 40.1529, 2.6186)

    # 25) <20 >30
    def test_diverge_20_30(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.closed(-p.inf, 20), p.closed(30, p.inf)]
            # is the negative std correct ?
            self.check_mean_std(y_intervals, 25, -1.2508)

    # interesting case only the lower 25 bound is taken into account
    # 26) <20 >25 >50 >50 >50
    def test_diverge_20_25_50_50_50(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.closed(-p.inf, 20), p.closed(25, p.inf), p.closed(50, p.inf), p.closed(50, p.inf), p.closed(50, p.inf)]
            self.check_mean_std(y_intervals, 22.5357, -0.9952)

    # 27) >30 35 37 <50
    def test_left_right_censoring_30_35_37_50(self):
        if ENABLE_LONG_RUNNING_TESTS:
            y_intervals = [p.closed(30, p.inf), p.singleton(35), p.singleton(37), p.closed(-p.inf, 50)]
            self.check_mean_std(y_intervals, 36.0000, 1.0001)

    # 28) <30 32 34 >50
    def test_divergent_30_32_34_50(self):
        y_intervals = [p.closed(-p.inf, 30), p.singleton(32), p.singleton(34), p.closed(50, p.inf)]
        self.check_mean_std(y_intervals, 36.0223, 14.4379)

    # 29) <10 <30 40 42 44
    def test_left_censored_data(self):
        y_intervals = [p.closed(-p.inf, 10), p.closed(-p.inf, 30), p.singleton(40), p.singleton(42), p.singleton(44)]
        self.check_mean_std(y_intervals, 28.0059, 18.9241)

    # 30) <10 <30 40 42 44 >60 >70
    def test_left_right_censoring_10_30_40_42_44_60_70(self):
        y_intervals = [p.closed(-p.inf, 10), p.closed(-p.inf, 30), p.singleton(40), p.singleton(42), p.singleton(44),
                p.closed(60, p.inf), p.closed(70, p.inf)]
        self.check_mean_std(y_intervals, 42.1983, 38.0563)