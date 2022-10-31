from experiments.synthetic.dynamic.dataset import *
from experiments.real.pm25.dataset import extract_features as extract_features_pm25, load_dataframe as load_dataframe_pm25, \
    numeric_features_column_names as numeric_features_pm25, x_numeric_fetures_mean as x_mean_pm25, x_numeric_fetures_std as x_std_pm25, pca
from experiments.real.bike_sharing.dataset import extract_features as extract_features_bike, load_dataframe as load_dataframe_bike, y_variable_label, \
    numeric_features_column_names as numeric_features_bike, x_numeric_fetures_mean as x_mean_bike, x_numeric_fetures_std as x_std_bike
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_distr
from experiments.synthetic.constants import ALPHA, BETA, NOISE
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


# PM25
def tresholds_from_percentiles_pm25(percentile_low, percentile_high, plot = False):
    df = load_dataframe_pm25()
    one_hot = sk.preprocessing.OneHotEncoder(sparse = False)
    month_one_hot = one_hot.fit_transform(np.expand_dims(df['month'].values, 1))
    hour_one_hot = one_hot.fit_transform(np.expand_dims(df['hour'].values, 1))
    combined_wind_direction_one_hot = one_hot.fit_transform(np.expand_dims(df['cbwd'].values, 1))
    numeric_fetures = df[numeric_features_pm25].values
    numeric_fetures = normalize(numeric_fetures, x_mean_pm25, x_std_pm25)
    x = np.hstack((month_one_hot, hour_one_hot, combined_wind_direction_one_hot, numeric_fetures))
    x = pca(x)
    y = df['pm2.5'].values
    bound_min, bound_max = np.percentile(y, [percentile_low, percentile_high])
    x_interval = [min(x), max(x)]
    if plot:
        plt.scatter(x, y, s = .1)
        plt.plot(x_interval, [bound_min] * 2, color = 'red', linewidth=.5)
        plt.plot(x_interval, [bound_max] * 2, color = 'red', linewidth=.5)
    return bound_min, bound_max

# tresholds_from_percentiles_pm25(10, 90, plot=True)



# Bike
def tresholds_from_percentiles_bike(percentile_low, percentile_high, plot = False):
    df = load_dataframe_bike()
    one_hot = sk.preprocessing.OneHotEncoder(sparse = False)
    hour_one_hot = one_hot.fit_transform(np.expand_dims(df['Hour'].values, 1))
    seansons_one_hot = one_hot.fit_transform(np.expand_dims(df['Seasons'].values, 1))
    holiday_one_hot = one_hot.fit_transform(np.expand_dims(df['Holiday'].values, 1))
    functioning_day_one_hot = one_hot.fit_transform(np.expand_dims(df['Functioning Day'].values, 1))
    # extract the numeric variables
    numeric_fetures = df[numeric_features_bike].values
    numeric_fetures = normalize(numeric_fetures, x_mean_bike, x_std_bike)
    # unite all features
    x = np.hstack((hour_one_hot, seansons_one_hot, holiday_one_hot, functioning_day_one_hot, numeric_fetures))
    x = pca(x)
    # extract the results
    y = df[y_variable_label].values
    bound_min, bound_max = np.percentile(y, [percentile_low, percentile_high])
    x_interval = [min(x), max(x)]
    if plot:
        plt.scatter(x, y, s = .1)
        plt.plot(x_interval, [bound_min] * 2, color = 'red', linewidth=.5)
        plt.plot(x_interval, [bound_max] * 2, color = 'red', linewidth=.5)
    return bound_min, bound_max

# tresholds_from_percentiles_bike(10, 90, plot=True)




