import matplotlib.pyplot as plt
import pandas as pd
from experiments.constants import LOSS, ABS_ERR, R_SQUARED
from experiments.real.pm25.dataset import extract_features, pca, bound_min, bound_max, zero_normalized, \
    CENSOR_LOW_BOUND, CENSOR_HIGH_BOUND, PM_2_5_Dataset
import numpy as np


def plot_full_dataset(df: pd.DataFrame, size = 0.01, label = None, censored = False):
    if censored:
        x, y = extract_features(df, lower_bound = CENSOR_LOW_BOUND, upper_bound = CENSOR_HIGH_BOUND)
    else:
        x, y = extract_features(df)
    x = pca(x)
    if not censored:
        min_max = [min(x), max(x)]
        plt.plot(min_max, [bound_min] * 2, color = 'red')
        plt.plot(min_max, [bound_max] * 2, color = 'red')
        plt.plot(min_max, [zero_normalized] * 2, color = 'red')
    plt.scatter(x, y, s = size, label = label)
    plt.xlabel('unidimensional PCA')
    plt.ylabel('PM2.5 (standardized)')
    # plt.savefig(label + '.pdf', dpi = 300, format = 'pdf')

def plot_train_test(train, test, title, y_title):
    plt.plot(range(len(train)), train, label = 'Train')
    plt.plot(range(len(test)), test, label = 'Test')
    plt.xlabel('Epochs')
    plt.ylabel(y_title)
    plt.title(title)
    plt.legend()
    # plt.savefig(title + '.png', dpi = 300, format = 'png')
    plt.show()

def plot_epochs(train_metrics_list, test_metrics_list):
    test_r2 = list(map(lambda m: m[R_SQUARED], test_metrics_list))
    test_err = list(map(lambda m: m[ABS_ERR], test_metrics_list))
    test_loss = list(map(lambda m: m[LOSS], test_metrics_list))
    train_r2 = list(map(lambda m: m[R_SQUARED], train_metrics_list))
    train_err = list(map(lambda m: m[ABS_ERR], train_metrics_list))
    train_loss = list(map(lambda m: m[LOSS], train_metrics_list))
    plot_train_test(train_loss, test_loss, 'Loss', 'Loss')
    plot_train_test(train_err, test_err, 'Absolute error', 'Absolute error')
    plot_train_test(train_r2, test_r2, 'R squared', 'R squared')

def plot_net(model, df: pd.DataFrame, sigma = None, label = 'model prediction', with_std = False):
    model.eval()
    x, y_real = extract_features(df)
    x_pca = pca(x)
    dataset = PM_2_5_Dataset(x, y_real)
    y_list = []
    for i in range(len(dataset)):
        x, _ = dataset[i]
        y_pred = model.forward(x.reshape(1, -1))
        y_list.append(y_pred[0].item())

    x_pca = np.squeeze(x_pca)
    np_y = np.array(y_list)
    indices_sorted = np.argsort(x_pca)

    x_pca_sorted = x_pca[indices_sorted]
    np_y_sorted = np_y[indices_sorted]

    if with_std and sigma:
      std = sigma.item()
      print('Std is ', std)
      plt.fill_between(x_pca_sorted, np_y_sorted + std, np_y_sorted - std, facecolor='gray', alpha=.6, label = 'Tobit std')

    plt.scatter(x_pca_sorted, np_y_sorted, s = .1, label = label)