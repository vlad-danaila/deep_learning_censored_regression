from implementation.util import to_numpy, to_torch
import torch as t
from implementation.normal_cumulative_distribution_function import cdf
from typing import Tuple

class TobitLoss(t.nn.Module):

    def __init__(self, device):
        super(TobitLoss, self).__init__()
        self.device = device
        self.gamma = to_torch(1, device = self.device, grad = True)

    def forward(self, x: Tuple[t.Tensor, t.Tensor, t.Tensor], y: Tuple[t.Tensor, t.Tensor, t.Tensor]):
        x_single_value, x_left_censored, x_right_censored = x
        y_single_value, y_left_censored, y_right_censored = y
        N = len(y_single_value) + len(y_left_censored) + len(y_right_censored)

        # Step 1: update based on pdf gradient (for uncensored data)
        # this is the same as -sum(ln(gamma) + ln(pdf(gamma * y - delta)))
        log_likelihood_pdf = to_torch(0, device = self.device, grad = True)
        if len(y_single_value) > 0:
            log_likelihood_pdf = -t.sum(t.log(self.gamma) - ((self.gamma * y_single_value - x_single_value) ** 2) / 2)

        # Step 2: compute the log(cdf(x)) gradient (for left censored data)
        log_likelihood_cdf = to_torch(0, device = self.device, grad = True)
        if len(y_left_censored) > 0:
            log_likelihood_cdf = -t.sum(t.log(cdf(self.gamma * y_left_censored - x_left_censored)))

        # Step 3: compute the log(1 - cdf(x)) = log(cdf(-x)) gradient (for right censored data)
        # notice the swaped signs for gamma and delta relations
        log_likelihood_1_minus_cdf = to_torch(0, device = self.device, grad = True)
        if len(y_right_censored) > 0:
            log_likelihood_1_minus_cdf = -t.sum(t.log(cdf(-self.gamma * y_right_censored + x_right_censored)))

        log_likelihood = log_likelihood_pdf + log_likelihood_cdf + log_likelihood_1_minus_cdf

        return log_likelihood

    # def mean(self):
    #     return self.delta / self.gamma

    def std(self):
        return 1 / self.gamma