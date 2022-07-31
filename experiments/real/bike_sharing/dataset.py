'''
This software uses the Seoul Bike Sharing Demand Data Set downloaded from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand

Dataset citation:

[1] Sathishkumar V E, Jangwoo Park, and Yongyun Cho. 'Using data mining techniques for bike sharing demand prediction in metropolitan city.' Computer Communications, Vol.153, pp.353-366, March, 2020

[2] Sathishkumar V E and Yongyun Cho. 'A rule-based model for Seoul Bike sharing demand prediction using weather data' European Journal of Remote Sensing, pp. 1-18, Feb, 2020

UCI Machine Learning Repository citation: re3data.org: UCI Machine Learning Repository; editing status 2017-10-30; re3data.org - Registry of Research Data Repositories. http://doi.org/10.17616/R3T91Q last accessed: 2020-12-20
'''

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

# URL_DATA_SET = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv'
DATASET_FILE = 'experiments/real/bike_sharing/dataset_bike_sharing.csv'
CENSOR_LOW_BOUND = 200
CENSOR_HIGH_BOUND = 2000
y_variable_label = 'Rented Bike Count'

INPUT_SIZE = 40

# r = requests.get(URL_DATA_SET, allow_redirects=True)
# open(DATASET_FILE, 'wb').write(r.content)

def load_dataframe(filter_null = True):
    df = pd.read_csv(DATASET_FILE)
    return df

df = load_dataframe()

train_samples = set(range(len(df)))
val_samples = set(range(2, len(df), 5))
test_samples = set(range(5, len(df), 5))
train_samples = train_samples - val_samples
train_samples = train_samples - test_samples

def train_df(df: pd.DataFrame):
    return df[ df.index.isin(train_samples) ]

def val_df(df: pd.DataFrame):
    return df[ df.index.isin(val_samples) ]

def test_df(df: pd.DataFrame):
    return df[ df.index.isin(test_samples) ]

def y_train_mean_std(max_iterations = 10_000, early_stop_patience = 5, lr = 1e-6, epsilon = 1e-6):
    df = train_df(load_dataframe())
    ys = df[y_variable_label].values

    real_mean, real_std = ys.mean(), ys.std()

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

numeric_features_column_names = ['Temperature(C)',	'Humidity(%)',	'Wind speed (m/s)',	'Visibility (10m)',	'Dew point temperature(C)',	'Solar Radiation (MJ/m2)',	'Rainfall(mm)',	'Snowfall (cm)']

def x_numeric_fatures_train_mean_std():
    df = train_df(load_dataframe())
    numeric_fetures = df[numeric_features_column_names].values
    mean, std = numeric_fetures.mean(axis = 0), numeric_fetures.std(axis = 0)
    return mean, std

x_numeric_fetures_mean, x_numeric_fetures_std = x_numeric_fatures_train_mean_std()

def extract_features(df: pd.DataFrame, lower_bound = -math.inf, upper_bound = math.inf):
    assert lower_bound <= upper_bound
    # handle categorical features (one hot encoding)
    one_hot = sk.preprocessing.OneHotEncoder(sparse = False)
    hour_one_hot = one_hot.fit_transform(np.expand_dims(df['Hour'].values, 1))
    seansons_one_hot = one_hot.fit_transform(np.expand_dims(df['Seasons'].values, 1))
    holiday_one_hot = one_hot.fit_transform(np.expand_dims(df['Holiday'].values, 1))
    functioning_day_one_hot = one_hot.fit_transform(np.expand_dims(df['Functioning Day'].values, 1))

    # extract the numeric variables
    numeric_fetures = df[numeric_features_column_names].values
    numeric_fetures = normalize(numeric_fetures, x_numeric_fetures_mean, x_numeric_fetures_std)

    # unite all features
    x = np.hstack((hour_one_hot, seansons_one_hot, holiday_one_hot, functioning_day_one_hot, numeric_fetures))

    # extract the results
    y = df[y_variable_label].values
    y = np.clip(y, lower_bound, upper_bound)
    y = normalize(y, y_mean, y_std)
    y = np.expand_dims(y, 1)

    return x, y

class Bike_Share_Dataset(t.utils.data.Dataset):

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

    dataset_train = Bike_Share_Dataset(*extract_features(df_train, lower_bound = CENSOR_LOW_BOUND, upper_bound = CENSOR_HIGH_BOUND))
    dataset_val = Bike_Share_Dataset(*extract_features(df_val, lower_bound = CENSOR_LOW_BOUND, upper_bound = CENSOR_HIGH_BOUND))
    dataset_test = Bike_Share_Dataset(*extract_features(df_test))

    return dataset_train, dataset_val, dataset_test

dataset_train, dataset_val, dataset_test = parse_datasets()

def pca(x, n_components = 1):
    pca_encoder = sk.decomposition.PCA(n_components = n_components)
    return pca_encoder.fit_transform(x)

n = len(dataset_train)
k = len(dataset_train[0][0])