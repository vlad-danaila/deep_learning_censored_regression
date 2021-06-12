import unittest
import torch as t
from censored_mae_mse.loss import censored_mse_with_truncation_penalty

class TobitOptimizationTest(unittest.TestCase):

    def test_uncensored(self):
        y = t.tensor([1.2, 1.5, 2])
        y_pred = t.tensor([1.3, 1.7, 2.3])
        expected_loss = t.tensor((.01 + .04 + .09) / 3)
        loss = censored_mse_with_truncation_penalty(y_pred = y_pred, y = y, clamp_all = True)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_lower_censored_1(self):
        low_censor = 1.4
        y = t.tensor([low_censor, 1.5, 2])
        y_pred = t.tensor([1.3, 1.7, 2.3])
        expected_loss = t.tensor((.04 + .09)/3)
        loss = censored_mse_with_truncation_penalty(y_pred = y_pred, y = y, lower_censoring_bound = low_censor, clamp_all = True)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_lower_censored_2(self):
        low_censor = 1.6
        y = t.tensor([low_censor, low_censor, 2])
        y_pred = t.tensor([1.3, 1.7, 2.3])
        expected_loss = t.tensor((.01 + .09)/3)
        loss = censored_mse_with_truncation_penalty(y_pred = y_pred, y = y, lower_censoring_bound = low_censor, clamp_all = True)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_lower_censored_3(self):
        low_censor = 1.6
        y = t.tensor([low_censor, 1.7, 2])
        y_pred = t.tensor([1.3, 1.0, 2.3])
        expected_loss = t.tensor((.01 + .09)/3)
        loss = censored_mse_with_truncation_penalty(y_pred = y_pred, y = y, lower_censoring_bound = low_censor, clamp_all = True)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_upper_censored_1(self):
        upper_censor = 1.6
        y = t.tensor([1.2, 1.5, upper_censor])
        y_pred = t.tensor([1.3, 1.7, 2.3])
        expected_loss = t.tensor((.01 + .01)/3)
        loss = censored_mse_with_truncation_penalty(y_pred = y_pred, y = y, upper_censoring_bound = upper_censor, clamp_all = True)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_upper_censored_2(self):
        upper_censor = 1.6
        y = t.tensor([1.2, upper_censor, upper_censor])
        y_pred = t.tensor([1.3, 1.5, 2.3])
        expected_loss = t.tensor((.01 + .01)/3)
        loss = censored_mse_with_truncation_penalty(y_pred = y_pred, y = y, upper_censoring_bound = upper_censor, clamp_all = True)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_upper_censored_3(self):
        upper_censor = 1.6
        y = t.tensor([1.2, 1.5, upper_censor])
        y_pred = t.tensor([1.3, 2.0, 2.3])
        expected_loss = t.tensor((.01 + .01)/3)
        loss = censored_mse_with_truncation_penalty(y_pred = y_pred, y = y, upper_censoring_bound = upper_censor, clamp_all = True)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_combined_1(self):
        lower_censor = 1.3
        upper_censor = 1.7
        y = t.tensor([lower_censor, lower_censor, 1.4, 1.5, upper_censor, upper_censor])
        y_pred = t.tensor([1.0, 1.4, 1.6, 2.0, 1.6, 1.8])
        expected_loss = t.tensor((0 + .01 + .04 + .04 + .01 + 0)/6)
        loss = censored_mse_with_truncation_penalty(y_pred = y_pred, y = y, lower_censoring_bound = lower_censor, upper_censoring_bound = upper_censor, clamp_all = True)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_truncated_1(self):
        lower_truncation = 1.
        y = t.tensor([1.1, 1.2, 1.3])
        y_pred = t.tensor([.9, 1.1, 1.3])
        expected_loss = t.tensor((.04 + .01 + .01 + 0)/3)
        loss = censored_mse_with_truncation_penalty(y_pred=y_pred, y=y, lower_truncation_limit = lower_truncation, clamp_all = True)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_truncated_2(self):
        lower_truncation = 1.
        y = t.tensor([1.1, 1.2, 1.3])
        y_pred = t.tensor([.9, .9, 1.3])
        expected_loss = t.tensor((.01 + .04 + .01 + 0.09 + 0)/3)
        loss = censored_mse_with_truncation_penalty(y_pred=y_pred, y=y, lower_truncation_limit = lower_truncation, clamp_all = True)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_truncated_3(self):
        upper_truncation = 1.
        y = t.tensor([.7, .8, .9])
        y_pred = t.tensor([.7, .9, 1.2])
        expected_loss = t.tensor((0 + 0.01 + 0.09 + 0.04)/3)
        loss = censored_mse_with_truncation_penalty(y_pred=y_pred, y=y, upper_truncation_limit = upper_truncation, clamp_all = True)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_combined_2(self):
        lower_censor = 1.3
        upper_censor = 1.7
        lower_truncation = 1.1
        upper_truncation = 1.8
        y = t.tensor([lower_censor, lower_censor, 1.4, 1.5, upper_censor, upper_censor])
        y_pred = t.tensor([1.0, 1.4, 1.6, 2.0, 1.6, 1.9])
        expected_loss = t.tensor((.01 + .01 + .04 + .04 + .04 + .01 + .01)/6)
        loss = censored_mse_with_truncation_penalty(y_pred = y_pred, y = y,
            lower_censoring_bound = lower_censor, upper_censoring_bound = upper_censor,
            lower_truncation_limit = lower_truncation, upper_truncation_limit = upper_truncation,
            clamp_all = True)
        self.assertTrue(t.allclose(loss, expected_loss))