import math
import numpy as np
from deep_tobit.util import to_torch, to_numpy, normalize, unnormalize
from scipy.stats import beta
from experiments.synthetic.constants import ALPHA, BETA, ABS_ERR, LOSS
import matplotlib.pyplot as plt


def plot_beta(x_mean, x_std, y_mean, y_std, lower = -math.inf, upper = math.inf, color = None, label = None, std = None):
    x = np.linspace(0, 1, 1000)
    beta_distribution = beta(a = ALPHA, b = BETA)
    y = beta_distribution.pdf(x)
    y = np.clip(y, lower, upper)
    x = normalize(x, x_mean, x_std)
    y = normalize(y, y_mean, y_std)
    plt.scatter(x, y, s = .1, color = color, label = label)
    if std:
        plt.fill_between(x, y + std, y - std, facecolor='blue', alpha=0.1, label = 'real std')

def plot_dataset(dataset, size = .01, label = None):
    x_list, y_list = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        x_list.append(x[0].item())
        y_list.append(y[0].item())
    plt.scatter(x_list, y_list, s = size, label = label)

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
    test_err = list(map(lambda m: m[ABS_ERR], test_metrics_list))
    test_loss = list(map(lambda m: m[LOSS], test_metrics_list))
    train_err = list(map(lambda m: m[ABS_ERR], train_metrics_list))
    train_loss = list(map(lambda m: m[LOSS], train_metrics_list))
    plot_train_test(train_loss, test_loss, 'Loss', 'Loss')
    plot_train_test(train_err, test_err, 'Absolute error', 'Absolute error')

def plot_net(model, dataset_val, start = 0, end = 1, gamma = None, label = 'model prediction', with_std = False):
    model.eval()
    x_list, y_list = [], []
    for i in range(len(dataset_val)):
        x, _ = dataset_val[i]
        y = model.forward(x.reshape(1, 1))
        if gamma:
            y = y / gamma
        x_list.append(x[0].item())
        y_list.append(y[0].item())
    plt.scatter(x_list, y_list, s = .1, label = label)
    if with_std and gamma:
        std = 1 / gamma.item()
        np_y = np.array(y_list)
        plt.fill_between(x_list, np_y + std, np_y - std, facecolor='gray', alpha=0.1, label = 'Tobit std')