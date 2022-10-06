from experiments.synthetic.dynamic.dataset import *
from experiments.util import TruncatedBetaDistributionConfig
from experiments.synthetic.constants import *
from experiments.synthetic.plot import plot_beta, plot_dataset

def plot_distribution(is_heteroscedastic, alpha, beta, censor_low_bound, censor_high_bound):
    config = TruncatedBetaDistributionConfig(
        censor_low_bound = censor_low_bound, censor_high_bound = censor_high_bound,
        alpha = alpha, beta = beta,
        is_heteroscedastic = is_heteroscedastic
    )
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(config)
    plot_beta(x_mean, x_std, y_mean, y_std, a=alpha, b=beta)
    plot_dataset(dataset_train)

def get_distirbution_variations():
    return [
        (False, 1, 4), (False, 2.5, 4), (False, 4, 4), (False, 4, 2.5), (False, 4, 1),
        (True, 1, 4), (True, 2.5, 4), (True, 4, 4), (True, 4, 2.5), (True, 4, 1)
    ]

plot_distribution(True, 1, 4, CENSOR_LOW_BOUND, CENSOR_HIGH_BOUND)