from experiments.synthetic.dynamic.dataset import *
from experiments.real.pm25.dataset import load_dataframe as load_dataframe_pm25, \
    numeric_features_column_names as numeric_features_pm25, x_numeric_fetures_mean as x_mean_pm25, x_numeric_fetures_std as x_std_pm25, pca
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_distr
from experiments.synthetic.constants import NOISE
import sklearn as sk

# Synthetic
def tresholds_from_percentiles_synthetic(is_heteroscedastic, alpha, beta, percentile_low, percentile_high, plot = False):
    nb_samples = 10_000

    x = np.linspace(0, 1, nb_samples)
    beta_distribution = beta_distr(a = alpha, b = beta)
    y = beta_distribution.pdf(x)
    noise = np.random.normal(0, NOISE, nb_samples)
    if is_heteroscedastic:
        noise = noise * y
    y += noise

    bound_min, bound_max = np.percentile(y, [percentile_low, percentile_high])

    if plot:
        plt.scatter(x, y, s = .1)
        plt.plot([0, 1], [bound_min] * 2, color = 'red', linewidth=.5)
        plt.plot([0, 1], [bound_max] * 2, color = 'red', linewidth=.5)

    return bound_min, bound_max

# is_heteroscedastic, alpha, beta = False, 1, 4
# percentile_low, percentile_high = 10, 90
# tresholds_from_percentiles_synthetic(is_heteroscedastic, alpha, beta, percentile_low, percentile_high, plot = True)

# Need to import them from datasets
# tresholds_from_percentiles_pm25(10, 90, plot=True)
# tresholds_from_percentiles_bike(10, 90, plot=True)




