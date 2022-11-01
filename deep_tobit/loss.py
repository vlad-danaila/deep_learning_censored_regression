from deep_tobit.util import to_torch
import torch as t
from deep_tobit.normal_cumulative_distribution_function import cdf
from typing import Tuple, Union

class Unscaled_Tobit_Loss(t.nn.Module):

    def __init__(self, device: Union[t.device, str, None], truncated_low: float = None, truncated_high: float = None, epsilon: float = 1e-10):
        super(Unscaled_Tobit_Loss, self).__init__()
        self.device = device
        self.truncated_low = truncated_low
        self.truncated_high = truncated_high
        self.epsilon = t.tensor(epsilon, dtype=t.float32, device=device, requires_grad=False)

    def forward(self, x: Tuple[t.Tensor, t.Tensor, t.Tensor], y: Tuple[t.Tensor, t.Tensor, t.Tensor]) -> t.Tensor:
        x_single_value, x_left_censored, x_right_censored = x
        y_single_value, y_left_censored, y_right_censored = y
        N = len(y_single_value) + len(y_left_censored) + len(y_right_censored)

        # Step 1: compute loss for uncensored data based on pdf:
        # -sum(ln(pdf(y - delta)))
        log_likelihood_pdf = to_torch(0, device = self.device, grad = True)
        if len(y_single_value) > 0:
            log_likelihood_pdf = t.sum(((y_single_value - x_single_value) ** 2) / 2)

        # Step 2: compute loss for left censored data:
        # -sum(ln(cdf(y - delta) - cdf(truncation - delta)))
        log_likelihood_cdf = to_torch(0, device = self.device, grad = True)
        if len(y_left_censored) > 0:
            truncation_low_penalty = 0 if not self.truncated_low else cdf(self.truncated_low - x_left_censored)
            log_likelihood_cdf = -t.sum(t.log(cdf(y_left_censored - x_left_censored) - truncation_low_penalty + self.epsilon))

        # Step 3: compute the loss for right censored data:
        # -sum(ln(cdf(delta - y) - cdf(delta - truncation)))
        # Notice that: log(1 - cdf(x)) = log(cdf(-x)), thus the swapped signs
        log_likelihood_1_minus_cdf = to_torch(0, device = self.device, grad = True)
        if len(y_right_censored) > 0:
            truncation_high_penalty = 0 if not self.truncated_high else cdf(x_right_censored - self.truncated_high)
            log_likelihood_1_minus_cdf = -t.sum(t.log(cdf(x_right_censored - y_right_censored) - truncation_high_penalty + self.epsilon))

        log_likelihood = log_likelihood_pdf + log_likelihood_cdf + log_likelihood_1_minus_cdf

        return log_likelihood

class Scaled_Tobit_Loss(t.nn.Module):

    def __init__(self, sigma: t.Tensor, device: Union[t.device, str, None], truncated_low: float = None, truncated_high: float = None, epsilon: float = 1e-10, std_penalty = None):
        super(Scaled_Tobit_Loss, self).__init__()
        self.sigma = sigma
        self.device = device
        self.truncated_low = truncated_low
        self.truncated_high = truncated_high
        self.epsilon = t.tensor(epsilon, dtype=t.float32, device=device, requires_grad=False)
        self.std_panalty = std_penalty

    def forward(self, x: Tuple[t.Tensor, t.Tensor, t.Tensor], y: Tuple[t.Tensor, t.Tensor, t.Tensor]) -> t.Tensor:
        x_single_value, x_left_censored, x_right_censored = x
        y_single_value, y_left_censored, y_right_censored = y
        N = len(y_single_value) + len(y_left_censored) + len(y_right_censored)

        sigma = t.abs(self.sigma)

        # Step 1: compute loss for uncensored data based on pdf:
        # -sum(ln(pdf((y - x)/sigma)) - ln(sigma))
        log_likelihood_pdf = to_torch(0, device = self.device, grad = True)
        if len(y_single_value) > 0:
            log_likelihood_pdf = -t.sum(-(((y_single_value - x_single_value) / sigma) ** 2) / 2 - t.log(sigma + self.epsilon))

        # Step 2: compute loss for left censored data:
        # -sum(ln(cdf((y - x)/sigma) - cdf((truncation - x)/sigma)))
        log_likelihood_cdf = to_torch(0, device = self.device, grad = True)
        if len(y_left_censored) > 0:
            truncation_low_penalty = 0 if not self.truncated_low else cdf((self.truncated_low - x_left_censored) / sigma)
            log_likelihood_cdf = -t.sum(t.log(cdf((y_left_censored - x_left_censored) / sigma) - truncation_low_penalty + self.epsilon))

        # Step 3: compute the loss for right censored data:
        # -sum(ln(cdf((delta - y)/sigma) - cdf((delta - truncation)/sigma)))
        # Notice that: log(1 - cdf(z)) = log(cdf(-z)), thus compared to step 2, the signs for gamma and x are swapped
        log_likelihood_1_minus_cdf = to_torch(0, device = self.device, grad = True)
        if len(y_right_censored) > 0:
            truncation_high_penalty = 0 if not self.truncated_high else cdf((-self.truncated_high + x_right_censored) / sigma)
            log_likelihood_1_minus_cdf = -t.sum(t.log(cdf((-y_right_censored + x_right_censored) / sigma) - truncation_high_penalty + self.epsilon))

        log_likelihood = log_likelihood_pdf + log_likelihood_cdf + log_likelihood_1_minus_cdf

        std_penalty = 0 if not self.std_panalty else self.std_panalty * sigma

        return log_likelihood + std_penalty

    def get_scale(self) -> t.Tensor:
        return t.abs(self.sigma)

class Reparametrized_Scaled_Tobit_Loss(t.nn.Module):

    def __init__(self, gamma: t.Tensor, device: Union[t.device, str, None], truncated_low: float = None, truncated_high: float = None, epsilon: float = 1e-10):
        super(Reparametrized_Scaled_Tobit_Loss, self).__init__()
        self.gamma = gamma
        self.device = device
        self.truncated_low = truncated_low
        self.truncated_high = truncated_high
        self.epsilon = t.tensor(epsilon, dtype=t.float32, device=device, requires_grad=False)

    def forward(self, x: Tuple[t.Tensor, t.Tensor, t.Tensor], y: Tuple[t.Tensor, t.Tensor, t.Tensor]) -> t.Tensor:
        x_single_value, x_left_censored, x_right_censored = x
        y_single_value, y_left_censored, y_right_censored = y
        N = len(y_single_value) + len(y_left_censored) + len(y_right_censored)

        gamma = t.abs(self.gamma)

        # Step 1: compute loss for uncensored data based on pdf:
        # -sum(ln(gamma) + ln(pdf(gamma * y - x)))
        log_likelihood_pdf = to_torch(0, device = self.device, grad = True)
        if len(y_single_value) > 0:
            log_likelihood_pdf = -t.sum(t.log(gamma + self.epsilon) - ((gamma * y_single_value - x_single_value) ** 2) / 2)

        # Step 2: compute loss for left censored data:
        # -sum(ln(cdf(gamma * y - x) - cdf(gamma * truncation - x)))
        log_likelihood_cdf = to_torch(0, device = self.device, grad = True)
        if len(y_left_censored) > 0:
            truncation_low_penalty = 0 if not self.truncated_low else cdf(gamma * self.truncated_low - x_left_censored)
            log_likelihood_cdf = -t.sum(t.log(cdf(gamma * y_left_censored - x_left_censored) - truncation_low_penalty + self.epsilon))

        # Step 3: compute the loss for right censored data:
        # -sum(ln(cdf(x - gamma * y) - cdf(x - gamma * truncation)))
        # Notice that: log(1 - cdf(z)) = log(cdf(-z)), thus compared to step 2, the signs for gamma and x are swapped
        log_likelihood_1_minus_cdf = to_torch(0, device = self.device, grad = True)
        if len(y_right_censored) > 0:
            truncation_high_penalty = 0 if not self.truncated_high else cdf(-gamma * self.truncated_high + x_right_censored)
            log_likelihood_1_minus_cdf = -t.sum(t.log(cdf(-gamma * y_right_censored + x_right_censored) - truncation_high_penalty + self.epsilon))

        log_likelihood = log_likelihood_pdf + log_likelihood_cdf + log_likelihood_1_minus_cdf

        return log_likelihood

    def get_scale(self) -> t.Tensor:
        return 1 / t.abs(self.gamma)

class Heteroscedastic_Reparametrized_Scaled_Tobit_Loss(t.nn.Module):

    def __init__(self, device: Union[t.device, str, None], truncated_low: float = None, truncated_high: float = None, epsilon: float = 1e-10):
        super(Heteroscedastic_Reparametrized_Scaled_Tobit_Loss, self).__init__()
        self.device = device
        self.truncated_low = truncated_low
        self.truncated_high = truncated_high
        self.epsilon = t.tensor(epsilon, dtype=t.float32, device=device, requires_grad=False)

    def forward(self, x: Tuple[t.Tensor, t.Tensor, t.Tensor], y: Tuple[t.Tensor, t.Tensor, t.Tensor], gamma: Tuple[t.Tensor, t.Tensor, t.Tensor]) -> t.Tensor:
        x_single_value, x_left_censored, x_right_censored = x
        y_single_value, y_left_censored, y_right_censored = y
        gamma_single_value, gamma_left_censored, gamma_right_censored = gamma
        gamma_single_value, gamma_left_censored, gamma_right_censored = t.abs(gamma_single_value), t.abs(gamma_left_censored), t.abs(gamma_right_censored)
        N = len(y_single_value) + len(y_left_censored) + len(y_right_censored)

        # Step 1: compute loss for uncensored data based on pdf:
        # -sum(ln(gamma) + ln(pdf(gamma * y - x)))
        log_likelihood_pdf = to_torch(0, device = self.device, grad = True)
        if len(y_single_value) > 0:
            log_likelihood_pdf = -t.sum(t.log(gamma_single_value + self.epsilon) - ((gamma_single_value * y_single_value - x_single_value) ** 2) / 2)

        # Step 2: compute loss for left censored data:
        # -sum(ln(cdf(gamma * y - x) - cdf(gamma * truncation - x)))
        log_likelihood_cdf = to_torch(0, device = self.device, grad = True)
        if len(y_left_censored) > 0:
            truncation_low_penalty = 0 if not self.truncated_low else cdf(gamma_left_censored * self.truncated_low - x_left_censored)
            log_likelihood_cdf = -t.sum(t.log(cdf(gamma_left_censored * y_left_censored - x_left_censored) - truncation_low_penalty + self.epsilon))

        # Step 3: compute the loss for right censored data:
        # -sum(ln(cdf(x - gamma * y) - cdf(x - gamma * truncation)))
        # Notice that: log(1 - cdf(z)) = log(cdf(-z)), thus compared to step 2, the signs for gamma and x are swapped
        log_likelihood_1_minus_cdf = to_torch(0, device = self.device, grad = True)
        if len(y_right_censored) > 0:
            truncation_high_penalty = 0 if not self.truncated_high else cdf(-gamma_right_censored * self.truncated_high + x_right_censored)
            log_likelihood_1_minus_cdf = -t.sum(t.log(cdf(-gamma_right_censored * y_right_censored + x_right_censored) - truncation_high_penalty + self.epsilon))

        log_likelihood = log_likelihood_pdf + log_likelihood_cdf + log_likelihood_1_minus_cdf

        return log_likelihood

class Heteroscedastic_Scaled_Tobit_Loss(t.nn.Module):

    def __init__(self, device: Union[t.device, str, None], truncated_low: float = None, truncated_high: float = None, epsilon: float = 1e-10, std_penalty = None):
        super(Heteroscedastic_Scaled_Tobit_Loss, self).__init__()
        self.device = device
        self.truncated_low = truncated_low
        self.truncated_high = truncated_high
        self.epsilon = t.tensor(epsilon, dtype=t.float32, device=device, requires_grad=False)
        self.std_panalty = std_penalty

    def forward(self, x: Tuple[t.Tensor, t.Tensor, t.Tensor], y: Tuple[t.Tensor, t.Tensor, t.Tensor], sigma: Tuple[t.Tensor, t.Tensor, t.Tensor]) -> t.Tensor:
        x_single_value, x_left_censored, x_right_censored = x
        y_single_value, y_left_censored, y_right_censored = y
        sigma_single_value, sigma_left_censored, sigma_right_censored = sigma
        sigma_single_value, sigma_left_censored, sigma_right_censored = t.abs(sigma_single_value), t.abs(sigma_left_censored), t.abs(sigma_right_censored)
        N = len(y_single_value) + len(y_left_censored) + len(y_right_censored)

        # Step 1: compute loss for uncensored data based on pdf:
        # -sum(ln(pdf((y - x)/sigma)) - ln(sigma))
        log_likelihood_pdf = to_torch(0, device = self.device, grad = True)
        if len(y_single_value) > 0:
            log_likelihood_pdf = -t.sum(-(((y_single_value - x_single_value) / sigma_single_value) ** 2) / 2 - t.log(sigma_single_value + self.epsilon))

        # Step 2: compute loss for left censored data:
        # -sum(ln(cdf((y - x)/sigma) - cdf((truncation - x)/sigma)))
        log_likelihood_cdf = to_torch(0, device = self.device, grad = True)
        if len(y_left_censored) > 0:
            truncation_low_penalty = 0 if not self.truncated_low else cdf((self.truncated_low - x_left_censored) / sigma_left_censored)
            log_likelihood_cdf = -t.sum(t.log(cdf((y_left_censored - x_left_censored) / sigma_left_censored) - truncation_low_penalty + self.epsilon))

        # Step 3: compute the loss for right censored data:
        # -sum(ln(cdf((delta - y)/sigma) - cdf((delta - truncation)/sigma)))
        # Notice that: log(1 - cdf(z)) = log(cdf(-z)), thus compared to step 2, the signs for sigma and x are swapped
        log_likelihood_1_minus_cdf = to_torch(0, device = self.device, grad = True)
        if len(y_right_censored) > 0:
            truncation_high_penalty = 0 if not self.truncated_high else cdf((-self.truncated_high + x_right_censored) / sigma_right_censored)
            log_likelihood_1_minus_cdf = -t.sum(t.log(cdf((-y_right_censored + x_right_censored) / sigma_right_censored) - truncation_high_penalty + self.epsilon))

        log_likelihood = log_likelihood_pdf + log_likelihood_cdf + log_likelihood_1_minus_cdf

        std_penalty = 0 if not self.std_panalty else self.std_panalty * ((t.sum(sigma_single_value) + t.sum(sigma_left_censored) + t.sum(sigma_right_censored)) / N)

        return log_likelihood + std_penalty