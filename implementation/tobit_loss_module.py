from implementation.util import to_numpy, to_torch
import torch as t
from scipy.stats import norm
import numpy as np
from implementation.util import normalize
from implementation.normal_cumulative_distribution_function import cdf

class TobitLoss(t.nn.Module):

    def __init__(self, device):
        super(TobitLoss, self).__init__()
        self.device = device
        self.delta = to_torch(0, device = self.device, grad = True)
        self.gamma = to_torch(1, device = self.device, grad = True)

    def forward(self, single_value, left_censored, right_censored):
        N = len(single_value) + len(left_censored) + len(right_censored)

        # Step 1: update based on pdf gradient (for uncensored data)
        # this is the same as -sum(ln(gamma) + ln(pdf(gamma * y - delta)))
        log_likelihood_pdf = to_torch(0, device = self.device, grad = True)
        if len(single_value) > 0:
            log_likelihood_pdf = -t.sum(t.log(self.gamma) - ((self.gamma * single_value - self.delta) ** 2)/2)

        # Step 2: compute the log(cdf(x)) gradient (for left censored data)
        log_likelihood_cdf = to_torch(0, device = self.device, grad = True)
        if len(left_censored) > 0:
            log_likelihood_cdf = -t.sum(t.log(cdf(self.gamma * left_censored - self.delta)))

        # Step 3: compute the log(1 - cdf(x)) = log(cdf(-x)) gradient (for right censored data)
        # notice the swaped signs for gamma and delta relations
        log_likelihood_1_minus_cdf = to_torch(0, device = self.device, grad = True)
        if len(right_censored) > 0:
            log_likelihood_1_minus_cdf = -t.sum(t.log(cdf(-self.gamma * right_censored + self.delta)))

        log_likelihood = log_likelihood_pdf + log_likelihood_cdf + log_likelihood_1_minus_cdf

        return log_likelihood

    def mean(self):
        return self.delta / self.gamma

    def std(self):
        return 1 / self.gamma

    def mean_std(self):
        return self.mean(), self.std()