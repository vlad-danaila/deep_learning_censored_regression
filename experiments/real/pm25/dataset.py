'''
This software uses the Beijing PM2.5 Data Data Set downloaded from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data

**Dataset citation:**
Liang, X., Zou, T., Guo, B., Li, S., Zhang, H., Zhang, S., Huang, H. and Chen, S. X. (2015). Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and winter heating. Proceedings of the Royal Society A, 471, 20150257.

**UCI Machine Learning Repository citation:**
re3data.org: UCI Machine Learning Repository; editing status 2017-10-30; re3data.org - Registry of Research Data Repositories. http://doi.org/10.17616/R3T91Q last accessed: 2020-12-20
'''

from matplotlib import pyplot as plt
import pandas as pd
import requests
import numpy as np
import math
from deep_tobit.util import normalize
import sklearn as sk
from experiments.constants import IS_CUDA_AVILABLE
import torch as t
import sklearn.preprocessing
import sklearn.decomposition

DATASET_FILE = 'experiments/real/pm25/Beijing PM2.5 dataset.csv'
URL_BEIJING_PM_2_5_DATA_SET = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv'

INPUT_SIZE = 46

r = requests.get(URL_BEIJING_PM_2_5_DATA_SET, allow_redirects=True)
open(DATASET_FILE, 'wb').write(r.content)

def load_dataframe(filter_null = True):
    df = pd.read_csv(DATASET_FILE)
    # exclude records without a measured pm2.5
    if filter_null:
        df = df[df['pm2.5'].notnull()]
    return df

df = load_dataframe()

def train_df(df: pd.DataFrame):
    return df[ df.year.isin([2010, 2011, 2012]) ]

def val_df(df: pd.DataFrame):
    return df[ df.year.isin([2013]) ]

def test_df(df: pd.DataFrame):
    return df[ df.year.isin([2014]) ]

numeric_features_column_names = ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']

def x_numeric_fatures_train_mean_std():
    df = train_df(load_dataframe())
    numeric_fetures = df[numeric_features_column_names].values
    mean, std = numeric_fetures.mean(axis = 0), numeric_fetures.std(axis = 0)
    return mean, std

x_numeric_fetures_mean, x_numeric_fetures_std = x_numeric_fatures_train_mean_std()

def pca(x, n_components = 1):
    pca_encoder = sk.decomposition.PCA(n_components = n_components)
    return pca_encoder.fit_transform(x)

def tresholds_from_percentiles_pm25(percentile_low, percentile_high, plot = False):
    df = load_dataframe()
    one_hot = sk.preprocessing.OneHotEncoder(sparse = False)
    month_one_hot = one_hot.fit_transform(np.expand_dims(df['month'].values, 1))
    hour_one_hot = one_hot.fit_transform(np.expand_dims(df['hour'].values, 1))
    combined_wind_direction_one_hot = one_hot.fit_transform(np.expand_dims(df['cbwd'].values, 1))
    numeric_fetures = df[numeric_features_column_names].values
    numeric_fetures = normalize(numeric_fetures, x_numeric_fetures_mean, x_numeric_fetures_std)
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

PERCENTILE = 0
c_low, c_high = tresholds_from_percentiles_pm25(PERCENTILE, 100 - PERCENTILE)
CENSOR_LOW_BOUND = c_low
CENSOR_HIGH_BOUND = c_high

def y_train_mean_std():
    df = train_df(load_dataframe())
    ys = df['pm2.5'].values

    y_single_valued, y_left_censored, y_right_censored = [], [], []

    for y in ys:
        if y > CENSOR_LOW_BOUND and y < CENSOR_HIGH_BOUND:
            y_single_valued.append(y)
        elif y <= CENSOR_LOW_BOUND:
            y_left_censored.append(CENSOR_LOW_BOUND)
        elif y >= CENSOR_HIGH_BOUND:
            y_right_censored.append(CENSOR_HIGH_BOUND)
        else:
            raise Exception('y outside of valid values, y = {}'.format(y[0]))

    all = np.array(y_single_valued + y_left_censored + y_right_censored)
    data_mean, data_std = all.mean(), all.std()

    return data_mean.item(), data_std.item()

y_mean, y_std = y_train_mean_std()
print(f'mean = {y_mean}; std = {y_std}')

bound_min = normalize(CENSOR_LOW_BOUND, y_mean, y_std)
bound_max = normalize(CENSOR_HIGH_BOUND, y_mean, y_std)
zero_normalized = normalize(0, y_mean, y_std)

def extract_features(df: pd.DataFrame, lower_bound = -math.inf, upper_bound = math.inf):
    assert lower_bound <= upper_bound
    # handle categorical features (one hot encoding)
    one_hot = sk.preprocessing.OneHotEncoder(sparse = False)
    month_one_hot = one_hot.fit_transform(np.expand_dims(df['month'].values, 1))
    # day_one_hot = one_hot.fit_transform(np.expand_dims(df['day'].values, 1))
    hour_one_hot = one_hot.fit_transform(np.expand_dims(df['hour'].values, 1))
    combined_wind_direction_one_hot = one_hot.fit_transform(np.expand_dims(df['cbwd'].values, 1))

    # extract the numeric variables
    numeric_fetures = df[numeric_features_column_names].values
    numeric_fetures = normalize(numeric_fetures, x_numeric_fetures_mean, x_numeric_fetures_std)

    # unite all features
    x = np.hstack((month_one_hot, hour_one_hot, combined_wind_direction_one_hot, numeric_fetures))

    # extract the results
    y = df['pm2.5'].values
    y = np.clip(y, lower_bound, upper_bound)
    y = normalize(y, y_mean, y_std)
    y = np.expand_dims(y, 1)

    return x, y

class PM_2_5_Dataset(t.utils.data.Dataset):

    def __init__(self, x: np.array, y: np.array, cuda = IS_CUDA_AVILABLE):
        super().__init__()
        self.x = t.tensor(x, requires_grad = True, dtype=t.float32)
        self.y = t.tensor(y, requires_grad = True, dtype=t.float32)
        if cuda:
            self.x = self.x.cuda()
            self.y = self.y.cuda()

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.y)

def parse_datasets():
    df = load_dataframe()

    # split into training / validation / test
    df_train = train_df(df)
    df_val = val_df(df)
    df_test = test_df(df)

    dataset_train = PM_2_5_Dataset(*extract_features(df_train, lower_bound = CENSOR_LOW_BOUND, upper_bound = CENSOR_HIGH_BOUND))
    dataset_val = PM_2_5_Dataset(*extract_features(df_val, lower_bound = CENSOR_LOW_BOUND, upper_bound = CENSOR_HIGH_BOUND))
    dataset_test = PM_2_5_Dataset(*extract_features(df_test))

    return dataset_train, dataset_val, dataset_test

dataset_train, dataset_val, dataset_test = parse_datasets()

n = len(dataset_train)
k = len(dataset_train[0][0])