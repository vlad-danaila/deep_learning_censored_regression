import math
import numpy as np
from scipy.stats import beta
from deep_tobit.util import normalize
import random
import torch as t
from experiments.synthetic.constants import ALPHA, BETA, NOISE
from experiments.synthetic.constants import DATASET_LEN
from experiments.util import get_device, set_random_seed, TruncatedBetaDistributionConfig


def calculate_mean_std(is_heteroscedastic, lower_bound = -math.inf, upper_bound = math.inf, nb_samples = DATASET_LEN, distribution_alpha = ALPHA, distribution_beta = BETA, start = 0, end = 1, noise = NOISE):
    assert lower_bound <= upper_bound
    beta_distribution = beta(a = distribution_alpha, b = distribution_beta)
    x = np.linspace(start, end, nb_samples)
    y = beta_distribution.pdf(x)
    noise = np.random.normal(0, noise, nb_samples)
    if is_heteroscedastic:
        noise = noise * y
    y += noise
    y = np.clip(y, lower_bound, upper_bound)
    return x.mean(), x.std(), y.mean(), y.std()

class TruncatedBetaDistributionDataset(t.utils.data.Dataset):

    def __init__(self, is_heteroscedastic, x_mean, x_std, y_mean, y_std, lower_bound = -math.inf, upper_bound = math.inf, nb_samples = DATASET_LEN, distribution_alpha = ALPHA, distribution_beta = BETA, noise = NOISE):
        super().__init__()
        assert lower_bound <= upper_bound
        self.is_heteroscedastic = is_heteroscedastic
        self.beta_distribution = beta(a = distribution_alpha, b = distribution_beta)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.noise = noise
        self.nb_samples = nb_samples
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

    def __getitem__(self, i):
        x = random.uniform(0, 1)
        y = self.beta_distribution.pdf(x)
        noise = random.gauss(0, self.noise)
        if self.is_heteroscedastic:
            noise = noise * y
        y += noise
        y = np.clip(y, self.lower_bound, self.upper_bound)
        x = normalize(x, mean = self.x_mean, std = self.x_std)
        y = normalize(y, mean = self.y_mean, std = self.y_std)
        return t.tensor([x], requires_grad = True, dtype=t.float32, device=get_device()), t.tensor([y], requires_grad = True, dtype=t.float32, device=get_device())

    def __len__(self):
        return self.nb_samples

class TruncatedBetaDistributionValidationDataset(TruncatedBetaDistributionDataset):

    def __init__(self, is_heteroscedastic, x_mean, x_std, y_mean, y_std, lower_bound = -math.inf, upper_bound = math.inf, nb_samples = DATASET_LEN, distribution_alpha = ALPHA, distribution_beta = BETA, start = 0, end = 1, noise = NOISE):
        super().__init__(is_heteroscedastic, x_mean, x_std, y_mean, y_std, lower_bound, upper_bound, nb_samples, distribution_alpha, distribution_beta)
        self.x = np.linspace(start, end, nb_samples)
        self.y = self.beta_distribution.pdf(self.x)
        noise = np.random.normal(0, noise, nb_samples)
        if is_heteroscedastic:
            noise = noise * self.y
        self.y += noise
        self.y = np.clip(self.y, self.lower_bound, self.upper_bound)
        self.x = normalize(self.x, mean = x_mean, std = x_std)
        self.y = normalize(self.y, mean = y_mean, std = y_std)
        self.x = np.expand_dims(self.x, axis = 1)
        self.y = np.expand_dims(self.y, axis = 1)
        self.x = t.tensor(self.x, requires_grad = False, dtype = t.float32, device=get_device())
        self.y = t.tensor(self.y, requires_grad = False, dtype = t.float32, device=get_device())

    def __getitem__(self, i):
        return self.x[i], self.y[i]

def get_experiment_data(dataset_config: TruncatedBetaDistributionConfig):
    # Reproducible experiments
    set_random_seed()
    # Configurations
    is_heteroscedastic = dataset_config.is_heteroscedastic
    low = dataset_config.censor_low_bound
    high = dataset_config.censor_high_bound
    alpha = dataset_config.alpha
    beta = dataset_config.beta
    # Mean / Std
    x_mean, x_std, y_mean, y_std = calculate_mean_std(
        is_heteroscedastic, lower_bound = low, upper_bound = high, distribution_alpha=alpha, distribution_beta=beta
    )
    # Datasets
    dataset_train = TruncatedBetaDistributionDataset(
        is_heteroscedastic, x_mean, x_std, y_mean, y_std,
        lower_bound = low, upper_bound = high,
        distribution_alpha=alpha, distribution_beta=beta
    )
    dataset_val = TruncatedBetaDistributionValidationDataset(
        is_heteroscedastic, x_mean, x_std, y_mean, y_std,
        lower_bound = low, upper_bound = high,
        distribution_alpha=alpha, distribution_beta=beta,
        nb_samples = 1000
    )
    dataset_test = TruncatedBetaDistributionValidationDataset(
        is_heteroscedastic, x_mean, x_std, y_mean, y_std,
        distribution_alpha=alpha, distribution_beta=beta,
    )
    # Normalization
    bound_min = normalize(low, y_mean, y_std)
    bound_max = normalize(high, y_mean, y_std)
    zero_normalized = normalize(0, y_mean, y_std)
    return dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std