from experiments.real.pm25.mse_based.mse_based_pm25 import bounded_loss
from experiments.synthetic.dynamic.dataset import *
from experiments.synthetic.eval_optimized import plot_and_evaluate_model_mae_mse, plot_dataset_and_net
from experiments.tpe_hyperparam_opt import get_objective_fn_mae_mse, tpe_opt_hyperparam
from experiments.util import TruncatedBetaDistributionConfig, name_from_distribution_config, create_folder
from experiments.synthetic.constants import *
from experiments.synthetic.plot import plot_beta, plot_dataset as plot_dataset_synt
from experiments.real.pm25.dataset import extract_features as extract_features_pm25, load_dataframe as load_dataframe_pm25
from experiments.real.pm25.plot import plot_full_dataset as plot_full_dataset_pm25
from experiments.real.bike_sharing.dataset import extract_features as extract_features_bike, load_dataframe as load_dataframe_bike
from experiments.real.bike_sharing.plot import plot_full_dataset as plot_full_dataset_bike
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_distr

# Synthetic
is_heteroscedastic, alpha, beta = False, 2.5, 4
config = TruncatedBetaDistributionConfig(
    censor_low_bound = CENSOR_LOW_BOUND, censor_high_bound = CENSOR_HIGH_BOUND,
    alpha = alpha, beta = beta,
    is_heteroscedastic = is_heteroscedastic
)
dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(config)
plot_dataset_synt(dataset_test)
config.censor_low_bound, config.censor_high_bound = np.percentile(dataset_test.y, [20, 80])
dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(config)
# plot_beta(x_mean, x_std, y_mean, y_std, a=alpha, b=beta)
plot_dataset_synt(dataset_train)




# PM25
df = load_dataframe_pm25()
x, y = extract_features_pm25(df)
y = y.reshape(y.shape[0])
bound_min, bound_max = np.percentile(y, [20, 80])
plot_full_dataset_pm25(df, bound_min=bound_min, bound_max=bound_max)

# Bike
df = load_dataframe_bike()
x, y = extract_features_bike(df)
y = y.reshape(y.shape[0])
bound_min, bound_max = np.percentile(y, [20, 80])
plot_full_dataset_bike(df, bound_min=bound_min, bound_max=bound_max)
