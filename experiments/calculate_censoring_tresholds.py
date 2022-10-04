from experiments.real.pm25.mse_based.mse_based_pm25 import bounded_loss
from experiments.synthetic.dynamic.dataset import *
from experiments.synthetic.eval_optimized import plot_and_evaluate_model_mae_mse, plot_dataset_and_net
from experiments.tpe_hyperparam_opt import get_objective_fn_mae_mse, tpe_opt_hyperparam
from experiments.util import TruncatedBetaDistributionConfig, name_from_distribution_config, create_folder
from experiments.synthetic.constants import *
from experiments.synthetic.plot import plot_beta, plot_dataset as plot_dataset_synt
from experiments.real.pm25.dataset import extract_features as extract_features_pm25, load_dataframe as load_dataframe_pm25, \
    numeric_features_column_names, x_numeric_fetures_mean, x_numeric_fetures_std, pca
from experiments.real.pm25.plot import plot_full_dataset as plot_full_dataset_pm25
from experiments.real.bike_sharing.dataset import extract_features as extract_features_bike, load_dataframe as load_dataframe_bike
from experiments.real.bike_sharing.plot import plot_full_dataset as plot_full_dataset_bike
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_distr
from experiments.synthetic.constants import ALPHA, BETA, NOISE
import sklearn as sk

# TODO: Find the unnormalized tresholds, because otherwise it impacts normalization

# Synthetic
def plot_tresholds_for_synthetic(is_heteroscedastic, alpha, beta, percentile_low, percentile_high):
    nb_samples = 10_000

    x = np.linspace(0, 1, nb_samples)
    beta_distribution = beta_distr(a = alpha, b = beta)
    y = beta_distribution.pdf(x)
    noise = np.random.normal(0, NOISE, nb_samples)
    if is_heteroscedastic:
        noise = noise * y
    y += noise

    bound_min, bound_max = np.percentile(y, [percentile_low, percentile_high])
    print(bound_min, bound_max)

    plt.scatter(x, y, s = .1)
    plt.plot([0, 1], [bound_min] * 2, color = 'red', linewidth=.5)
    plt.plot([0, 1], [bound_max] * 2, color = 'red', linewidth=.5)

is_heteroscedastic, alpha, beta = False, 2.5, 4
percentile_low, percentile_high = 20, 80
plot_tresholds_for_synthetic(is_heteroscedastic, alpha, beta, percentile_low, percentile_high)




# PM25
# TODO
df = load_dataframe_pm25()
one_hot = sk.preprocessing.OneHotEncoder(sparse = False)
month_one_hot = one_hot.fit_transform(np.expand_dims(df['month'].values, 1))
hour_one_hot = one_hot.fit_transform(np.expand_dims(df['hour'].values, 1))
combined_wind_direction_one_hot = one_hot.fit_transform(np.expand_dims(df['cbwd'].values, 1))
numeric_fetures = df[numeric_features_column_names].values
numeric_fetures = normalize(numeric_fetures, x_numeric_fetures_mean, x_numeric_fetures_std)
x = np.hstack((month_one_hot, hour_one_hot, combined_wind_direction_one_hot, numeric_fetures))
x = pca(x)
y = df['pm2.5'].values
plt.scatter(x, y, s = .1)




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
