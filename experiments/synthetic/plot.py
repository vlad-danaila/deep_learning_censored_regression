import math
import numpy as np
from deep_tobit.util import to_numpy, normalize
from scipy.stats import beta
from experiments.synthetic.constants import ALPHA, BETA
from experiments.constants import LOSS, ABS_ERR, DOT_SIZE
import matplotlib.pyplot as plt
import torch as t
from experiments.util import scatterplot

def plot_beta(x_mean, x_std, y_mean, y_std, lower = -math.inf, upper = math.inf, label = None, std = None):
    x = np.linspace(0, 1, 1000)
    beta_distribution = beta(a = ALPHA, b = BETA)
    y = beta_distribution.pdf(x)
    y = np.clip(y, lower, upper)
    x = normalize(x, x_mean, x_std)
    y = normalize(y, y_mean, y_std)
    scatterplot(x, y, label = label)
    if std:
        plt.fill_between(x, y + std, y - std, facecolor='blue', alpha=0.1, label = 'real std')

def plot_dataset(dataset, size = DOT_SIZE, label = None):
    x_list, y_list = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        x_list.append(x[0].item())
        y_list.append(y[0].item())
    scatterplot(x_list, y_list, label = label)

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

def plot_net(model, dataset_val, sigma = None, gamma = None, sigma_model = None, gamma_model = None, label = 'model prediction', with_std = False):
    model.eval()
    if sigma_model is not None:
        sigma_model.eval()
    if gamma_model is not None:
        gamma_model.eval()
    x_list, y_list = [], []
    for i in range(len(dataset_val)):
        x, _ = dataset_val[i]
        y = model.forward(x.reshape(1, 1))
        if gamma:
            y = y / gamma
        elif gamma_model is not None:
            y = y / t.abs(gamma_model.forward(x.reshape(-1, 1)))
        x_list.append(x[0].item())
        y_list.append(y[0].item())
    if with_std and gamma:
        std = 1 / gamma.item()
        np_y = np.array(y_list)
        plt.fill_between(x_list, np_y + std, np_y - std, facecolor='gray', alpha=0.1, label = 'Tobit std')
    elif with_std and sigma:
        std = sigma.item()
        np_y = np.array(y_list)
        plt.fill_between(x_list, np_y + std, np_y - std, facecolor='gray', alpha=0.1, label = 'Tobit std')
    elif with_std and sigma_model is not None:
        x_list = np.array(x_list).squeeze()
        np_y = np.array(y_list).squeeze()
        std = to_numpy(t.abs(sigma_model(t.tensor(x_list.reshape(-1, 1), dtype=t.float32))))
        std = std.squeeze()
        plt.fill_between(x_list, np_y + std, np_y - std, facecolor='gray', alpha=0.1, label = 'Tobit std')
    elif with_std and gamma_model is not None:
        x_list = np.array(x_list).squeeze()
        np_y = np.array(y_list).squeeze()
        std = to_numpy(1 / t.abs(gamma_model(t.tensor(x_list.reshape(-1, 1), dtype=t.float32))))
        std = std.squeeze()
        plt.fill_between(x_list, np_y + std, np_y - std, facecolor='gray', alpha=0.1, label = 'Tobit std')
    scatterplot(x_list, y_list, label = label)

def plot_fixed_and_dynamic_std(dataset_val, model, scale_model, fixed_scale, is_reparam = False):
    scale_model.eval()
    x_list, y_list = [], []
    for i in range(len(dataset_val)):
        x, _ = dataset_val[i]
        x_list.append(x[0].item())
    x_list = np.array(x_list).squeeze()
    if is_reparam:
        std = to_numpy(1 / t.abs(scale_model(t.tensor(x_list.reshape(-1, 1), dtype=t.float32))))
    else:
        std = to_numpy(t.abs(scale_model(t.tensor(x_list.reshape(-1, 1), dtype=t.float32))))
    std = std.squeeze()
    plt.plot(x_list, std, label = 'dynamic std', linewidth = 1)
    plt.plot(x_list, [fixed_scale] * len(x_list), label ='fixed std', linewidth = 1)