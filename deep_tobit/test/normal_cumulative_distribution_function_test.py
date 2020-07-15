from deep_tobit.util import to_torch, to_numpy, normalize
import torch as t
from scipy.stats import norm
import unittest
import numpy as np
from deep_tobit.normal_cumulative_distribution_function import cdf
from numpy.testing import assert_almost_equal

class CDFTest(unittest.TestCase):

    def test_cdf_gradient(self):
        input = [10, 15, 20, 25, 30]

        # manual gradient computing
        x = np.array(input)
        mean, std = x.mean(), x.std()
        x_normalized = normalize(x, mean, std)
        expected_cdf = norm.cdf(x_normalized)
        expected_log_likelihood = np.log(expected_cdf)
        expected_grad_log_likelihood_by_x = norm.pdf(x_normalized) / (expected_cdf * std)

        # automatic gradient computing
        x = to_torch(input, grad = True)
        # in this test mean & std are considered constants
        x_normalized = normalize(x, mean, std)
        cdf_result = cdf(x_normalized)
        assert_almost_equal(to_numpy(cdf_result), expected_cdf)

        log_likelihood_result = t.log(cdf_result)
        assert_almost_equal(to_numpy(log_likelihood_result), expected_log_likelihood)

        loss = t.sum(log_likelihood_result)
        loss.backward()
        assert_almost_equal(to_numpy(x.grad), expected_grad_log_likelihood_by_x)