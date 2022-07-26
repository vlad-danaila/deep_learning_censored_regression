from experiments.synthetic.constants import CENSOR_LOW_BOUND, CENSOR_HIGH_BOUND
from experiments.synthetic.constant_noise.dataset import calculate_mean_std
from experiments.synthetic.constants import DATASET_LEN, LOSS, ABS_ERR, R_SQUARED, CHECKPOINT_FREQUENCY
import numpy as np
import torch as t
import sklearn as sk
from deep_tobit.util import to_torch, to_numpy, normalize, unnormalize
import math
import traceback

# n is the nb of samples, k is the nb of regressors (features)
n = DATASET_LEN
k = 1 # univariate

def adjusted_R2(y, y_pred):
    r2 = sk.metrics.r2_score(y, y_pred)
    return 1 - ( ( (1 - r2) * (n - 1) ) / (n - k - 1) )

def eval_network_mae_mse_gll(bound_min, bound_max, model, loader, loss_fn, batch_size, is_eval_bounded = True):
    model.eval()
    with t.no_grad():
        metrics = np.zeros(3)
        total_weight = 0
        for x, y in loader:
            y_pred = model.forward(x)
            loss = loss_fn(y_pred, y)
            if len(y_pred) == 3:
                y_pred = t.cat(y_pred)
                y = t.cat(y)
            if hasattr(loss_fn, 'gamma'):
                y_pred = y_pred / (loss_fn.gamma)
            if is_eval_bounded:
                y_pred = t.clamp(y_pred, min = bound_min, max = bound_max)
            y_pred, y = to_numpy(y_pred), to_numpy(y)
            weight = len(y) / batch_size
            metrics[LOSS] += (loss.item() * weight)
            metrics[ABS_ERR] += (sk.metrics.mean_absolute_error(y, y_pred) * weight)
            metrics[R_SQUARED] += (adjusted_R2(y, y_pred) * weight)
            total_weight += weight
        metrics /= total_weight
        return metrics

def eval_network_tobit(bound_min, bound_max, model, loader, loss_fn, batch_size, is_eval_bounded = True):
    model.eval()
    with t.no_grad():
        metrics = np.zeros(3)
        total_weight = 0
        for x, y, single_valued_indexes, left_censored_indexes, right_censored_indexes in loader:
            y_single_valued = y[single_valued_indexes]
            y_left_censored = y[left_censored_indexes]
            y_right_censored = y[right_censored_indexes]
            y_tuple = y_single_valued, y_left_censored, y_right_censored
            y_pred = model.forward(x)
            y_pred_single_valued = y_pred[single_valued_indexes]
            y_pred_left_censored = y_pred[left_censored_indexes]
            y_pred_right_censored = y_pred[right_censored_indexes]
            y_pred_tuple = y_pred_single_valued, y_pred_left_censored, y_pred_right_censored
            loss = loss_fn(y_pred_tuple, y_tuple)
            if hasattr(loss_fn, 'gamma'):
                y_pred = y_pred / (loss_fn.gamma)
            if is_eval_bounded:
                y_pred = t.clamp(y_pred, min = bound_min, max = bound_max)
            y_pred, y = to_numpy(y_pred), to_numpy(y)
            weight = len(y) / batch_size
            metrics[LOSS] += (loss.item() * weight)
            metrics[ABS_ERR] += (sk.metrics.mean_absolute_error(y, y_pred) * weight)
            metrics[R_SQUARED] += (adjusted_R2(y, y_pred) * weight)
            total_weight += weight
        metrics /= total_weight
        return metrics

def train_network_mae_mse_gll(bound_min, bound_max, model, loss_fn, optimizer, scheduler, loader_train, loader_val, checkpoint_name, batch_size_train, batch_size_val, epochs, log = True):
    metrics_train_per_epochs, metrics_test_per_epochs = [], []
    best = [math.inf, math.inf, -math.inf]
    try:
        counter = 0
        total_weight = 0
        train_metrics = np.zeros(3)
        for epoch in range(epochs):
            try:
                model.train()
                for x, y in loader_train:
                    counter += 1
                    y_pred = model.forward(x)
                    loss = loss_fn(y_pred, y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    if hasattr(loss_fn, 'gamma'):
                        y_pred = y_pred / (loss_fn.gamma)
                    y_pred = t.clamp(y_pred, min = bound_min, max = bound_max)
                    y_pred, y = to_numpy(y_pred), to_numpy(y)
                    weight = len(y) / batch_size_train
                    train_metrics[LOSS] += (loss.item() * weight)
                    train_metrics[ABS_ERR] += (sk.metrics.mean_absolute_error(y, y_pred) * weight)
                    train_metrics[R_SQUARED] += (adjusted_R2(y, y_pred) * weight)
                    total_weight += weight
                    scheduler.step()
                    if counter % CHECKPOINT_FREQUENCY == 0:
                        train_metrics /= total_weight
                        metrics_train_per_epochs.append(train_metrics)
                        train_metrics = np.zeros(3)
                        total_weight = 0
                        test_metrics = eval_network_mae_mse_gll(bound_min, bound_max, model, loader_val, loss_fn, batch_size_val)
                        metrics_test_per_epochs.append(test_metrics)
                        # if test_metrics[R_SQUARED] > best[R_SQUARED]:
                        if test_metrics[ABS_ERR] < best[ABS_ERR]:
                            best = test_metrics
                            checkpoint_dict = {'model': model.state_dict()}
                            if hasattr(loss_fn, 'gamma'):
                                checkpoint_dict['gamma'] = loss_fn.gamma
                            if hasattr(loss_fn, 'sigma'):
                                checkpoint_dict['sigma'] = loss_fn.sigma
                            t.save(checkpoint_dict, '{}.tar'.format(checkpoint_name))
                        if log:
                            print('Iteration {} abs err {} R2 {}'.format(counter, test_metrics[ABS_ERR], test_metrics[R_SQUARED]))
            except Exception as e:
                print(traceback.format_exc())
                break
        print('Best absolute error:', best[ABS_ERR], 'R2:', best[R_SQUARED])
        return metrics_train_per_epochs, metrics_test_per_epochs, best
    except KeyboardInterrupt as e:
        print('Training interrupted at epoch', epoch)

def train_network_tobit(bound_min, bound_max, model, loss_fn, optimizer, scheduler, loader_train, loader_val, checkpoint_name, batch_size_train, batch_size_val, epochs, log = True):
    metrics_train_per_epochs, metrics_test_per_epochs = [], []
    best = [math.inf, math.inf, -math.inf]
    try:
        counter = 0
        total_weight = 0
        train_metrics = np.zeros(3)
        for epoch in range(epochs):
            try:
                model.train()
                for x, y, single_valued_indexes, left_censored_indexes, right_censored_indexes in loader_train:
                    counter += 1
                    y_single_valued = y[single_valued_indexes]
                    y_left_censored = y[left_censored_indexes]
                    y_right_censored = y[right_censored_indexes]
                    y_tuple = y_single_valued, y_left_censored, y_right_censored
                    y_pred = model.forward(x)
                    y_pred_single_valued = y_pred[single_valued_indexes]
                    y_pred_left_censored = y_pred[left_censored_indexes]
                    y_pred_right_censored = y_pred[right_censored_indexes]
                    y_pred_tuple = y_pred_single_valued, y_pred_left_censored, y_pred_right_censored
                    loss = loss_fn(y_pred_tuple, y_tuple)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    if hasattr(loss_fn, 'gamma'):
                        y_pred = y_pred / (loss_fn.gamma)
                    y_pred = t.clamp(y_pred, min = bound_min, max = bound_max)
                    y_pred, y = to_numpy(y_pred), to_numpy(y)
                    weight = len(y) / batch_size_train
                    train_metrics[LOSS] += (loss.item() * weight)
                    train_metrics[ABS_ERR] += (sk.metrics.mean_absolute_error(y, y_pred) * weight)
                    train_metrics[R_SQUARED] += (adjusted_R2(y, y_pred) * weight)
                    total_weight += weight
                    scheduler.step()
                    if counter % CHECKPOINT_FREQUENCY == 0:
                        train_metrics /= total_weight
                        metrics_train_per_epochs.append(train_metrics)
                        train_metrics = np.zeros(3)
                        total_weight = 0
                        test_metrics = eval_network_tobit(bound_min, bound_max, model, loader_val, loss_fn, batch_size_val)
                        metrics_test_per_epochs.append(test_metrics)
                        # if test_metrics[R_SQUARED] > best[R_SQUARED]:
                        if test_metrics[ABS_ERR] < best[ABS_ERR]:
                            best = test_metrics
                            checkpoint_dict = {'model': model.state_dict()}
                            if hasattr(loss_fn, 'gamma'):
                                checkpoint_dict['gamma'] = loss_fn.gamma
                            if hasattr(loss_fn, 'sigma'):
                                checkpoint_dict['sigma'] = loss_fn.sigma
                            t.save(checkpoint_dict, '{}.tar'.format(checkpoint_name))
                        if log:
                            print('Iteration {} abs err {} R2 {}'.format(counter, test_metrics[ABS_ERR], test_metrics[R_SQUARED]))
            except Exception as e:
                print(traceback.format_exc())
                break
        print('Best absolute error:', best[ABS_ERR], 'R2:', best[R_SQUARED])
        return metrics_train_per_epochs, metrics_test_per_epochs, best
    except KeyboardInterrupt as e:
        print('Training interrupted at epoch', epoch)