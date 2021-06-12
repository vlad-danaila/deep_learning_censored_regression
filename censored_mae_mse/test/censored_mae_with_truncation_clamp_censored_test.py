import unittest
import torch as t
from censored_mae_mse.loss import censored_mae_with_truncation_penalty

class TobitOptimizationTest(unittest.TestCase):

    def test_uncensored(self):
        y = t.tensor([1.2, 1.5, 2])
        y_pred = t.tensor([1.3, 1.7, 2.3])
        expected_loss = t.tensor(.2)
        loss = censored_mae_with_truncation_penalty(y_pred = y_pred, y = y)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_lower_censored_1(self):
        low_censor = 1.4
        y = t.tensor([low_censor, 1.5, 2])
        y_pred = t.tensor([1.3, 1.7, 2.3])
        expected_loss = t.tensor((.2 + .3)/3)
        loss = censored_mae_with_truncation_penalty(y_pred = y_pred, y = y, lower_censoring_bound = low_censor)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_lower_censored_2(self):
        low_censor = 1.6
        y = t.tensor([low_censor, low_censor, 2])
        y_pred = t.tensor([1.3, 1.7, 2.3])
        expected_loss = t.tensor((.1 + .3)/3)
        loss = censored_mae_with_truncation_penalty(y_pred = y_pred, y = y, lower_censoring_bound = low_censor)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_lower_censored_3(self):
        low_censor = 1.6
        y = t.tensor([low_censor, 1.7, 2])
        y_pred = t.tensor([1.3, 1.0, 2.3])
        expected_loss = t.tensor((.7 + .3)/3)
        loss = censored_mae_with_truncation_penalty(y_pred = y_pred, y = y, lower_censoring_bound = low_censor)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_upper_censored_1(self):
        upper_censor = 1.6
        y = t.tensor([1.2, 1.5, upper_censor])
        y_pred = t.tensor([1.3, 1.7, 2.3])
        expected_loss = t.tensor((.1 + .2)/3)
        loss = censored_mae_with_truncation_penalty(y_pred = y_pred, y = y, upper_censoring_bound = upper_censor)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_upper_censored_2(self):
        upper_censor = 1.6
        y = t.tensor([1.2, upper_censor, upper_censor])
        y_pred = t.tensor([1.3, 1.5, 2.3])
        expected_loss = t.tensor((.1 + .1)/3)
        loss = censored_mae_with_truncation_penalty(y_pred = y_pred, y = y, upper_censoring_bound = upper_censor)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_upper_censored_3(self):
        upper_censor = 1.6
        y = t.tensor([1.2, 1.5, upper_censor])
        y_pred = t.tensor([1.3, 2.0, 2.3])
        expected_loss = t.tensor((.1 + .5)/3)
        loss = censored_mae_with_truncation_penalty(y_pred = y_pred, y = y, upper_censoring_bound = upper_censor)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_combined_1(self):
        lower_censor = 1.3
        upper_censor = 1.7
        y = t.tensor([lower_censor, lower_censor, 1.4, 1.5, upper_censor, upper_censor])
        y_pred = t.tensor([1.0, 1.4, 1.6, 2.0, 1.6, 1.8])
        expected_loss = t.tensor((0 + .1 + .2 + .5 + .1 + 0)/6)
        loss = censored_mae_with_truncation_penalty(y_pred = y_pred, y = y, lower_censoring_bound = lower_censor, upper_censoring_bound = upper_censor)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_truncated_1(self):
        lower_truncation = 1.
        y = t.tensor([1.1, 1.2, 1.3])
        y_pred = t.tensor([.9, 1.1, 1.3])
        expected_loss = t.tensor((0.3 + 0.1 + 0)/3)
        loss = censored_mae_with_truncation_penalty(y_pred=y_pred, y=y, lower_truncation_limit = lower_truncation)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_truncated_2(self):
        lower_truncation = 1.
        y = t.tensor([1.1, 1.2, 1.3])
        y_pred = t.tensor([.9, .9, 1.3])
        expected_loss = t.tensor((0.3 + 0.4 + 0)/3)
        loss = censored_mae_with_truncation_penalty(y_pred=y_pred, y=y, lower_truncation_limit = lower_truncation)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_truncated_3(self):
        upper_truncation = 1.
        y = t.tensor([.7, .8, .9])
        y_pred = t.tensor([.7, .9, 1.2])
        expected_loss = t.tensor((0 + 0.1 + 0.5)/3)
        loss = censored_mae_with_truncation_penalty(y_pred=y_pred, y=y, upper_truncation_limit = upper_truncation)
        self.assertTrue(t.allclose(loss, expected_loss))

    def test_combined_2(self):
        lower_censor = 1.3
        upper_censor = 1.7
        lower_truncation = 1.1
        upper_truncation = 1.8
        y = t.tensor([lower_censor, lower_censor, 1.4, 1.5, upper_censor, upper_censor])
        y_pred = t.tensor([1.0, 1.4, 1.6, 2.0, 1.6, 1.9])
        expected_loss = t.tensor((.1 + .1 + .2 + .7 + .1 + .1)/6)
        loss = censored_mae_with_truncation_penalty(y_pred = y_pred, y = y,
            lower_censoring_bound = lower_censor, upper_censoring_bound = upper_censor, lower_truncation_limit = lower_truncation, upper_truncation_limit = upper_truncation)
        self.assertTrue(t.allclose(loss, expected_loss))