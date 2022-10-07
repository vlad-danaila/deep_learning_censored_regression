from experiments.synthetic.dynamic.dataset import *
from experiments.synthetic.plot import plot_beta, plot_dataset
from experiments.util import TruncatedBetaDistributionConfig
from experiments.calculate_censoring_tresholds import tresholds_from_percentiles_synthetic

def plot_distribution(is_heteroscedastic, alpha, beta, percentile_low, percentile_high):
    censor_low_bound, censor_high_bound = tresholds_from_percentiles_synthetic(is_heteroscedastic, alpha, beta, percentile_low, percentile_high)
    config = TruncatedBetaDistributionConfig(
        censor_low_bound = censor_low_bound, censor_high_bound = censor_high_bound,
        alpha = alpha, beta = beta,
        is_heteroscedastic = is_heteroscedastic
    )
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(config)
    plot_beta(x_mean, x_std, y_mean, y_std, a=alpha, b=beta)
    plot_dataset(dataset_train)

def get_distirbution_variations() -> TruncatedBetaDistributionConfig:
    beta = 4
    for is_heteroscedastic in [False, True]:
        for alpha in [1, 2, 4]:
            for percentile_low in [0, 10, 20, 30]:
                percentile_high = 100 - percentile_low
                censor_low_bound, censor_high_bound = tresholds_from_percentiles_synthetic(
                    is_heteroscedastic, alpha, beta, percentile_low, percentile_high
                )
                yield TruncatedBetaDistributionConfig(
                    censor_low_bound, censor_high_bound, alpha, beta, is_heteroscedastic
                )


# plot_distribution(False, 1, 4, 20, 80)