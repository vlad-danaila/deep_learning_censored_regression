import math
import matplotlib.pyplot as plt
import torch as t
from deep_tobit.loss import Reparametrized_Scaled_Tobit_Loss, Scaled_Tobit_Loss, \
    Heteroscedastic_Reparametrized_Scaled_Tobit_Loss, Heteroscedastic_Scaled_Tobit_Loss
from deep_tobit.util import distinguish_censored_versus_observed_data

from experiments.constants import ABS_ERR, R_SQUARED
from experiments.synthetic.models import DenseNetwork, get_scale_network
from experiments.synthetic.plot import plot_beta, plot_dataset, plot_net
from experiments.train import eval_network_mae_mse_gll, eval_network_tobit_fixed_std, eval_network_tobit_dyn_std
from experiments.util import save_fig_in_checkpoint_folder, get_device
from experiments.models import get_dense_net

def plot_dataset_and_net(checkpoint, model, x_mean, x_std, y_mean, y_std, dataset_val, with_std=False, scale_model=None):
    model.load_state_dict(checkpoint['model'])
    plot_beta(x_mean, x_std, y_mean, y_std, label = 'true distribution')
    plot_dataset(dataset_val, label = 'validation data')
    if 'sigma' in checkpoint:
        if scale_model:
            plot_net(model, dataset_val, sigma_model = scale_model, with_std = with_std)
        else:
            plot_net(model, dataset_val, sigma = checkpoint['sigma'], with_std = with_std)
    elif 'gamma' in checkpoint:
        if scale_model:
            plot_net(model, dataset_val, gamma_model = scale_model, with_std = with_std)
        else:
            plot_net(model, dataset_val, gamma = checkpoint['gamma'], with_std = with_std)
    else:
        plot_net(model, dataset_val)
    plt.xlabel('input (standardized)')
    plt.ylabel('outcome (standardized)')
    plt.ylim((-2.5, 2.5))
    lgnd = plt.legend()
    lgnd.legendHandles[0]._sizes = [10]
    lgnd.legendHandles[1]._sizes = [10]
    lgnd.legendHandles[2]._sizes = [10]
    if with_std:
        lgnd.legendHandles[3]._sizes = [10]

def plot_and_evaluate_model_mae_mse(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test, root_folder,
                                    checkpoint_name, criterion, is_optimized = True, input_size = 1, is_liniar = False, loader_val = None):
    model = model_fn()
    loss_fn = criterion()
    checkpoint = t.load(root_folder + '/' + checkpoint_name + (' best.tar' if is_optimized else '.tar'))
    plot_dataset_and_net(checkpoint, model, x_mean, x_std, y_mean, y_std, dataset_val)
    save_fig_in_checkpoint_folder(root_folder, checkpoint_name)

    if not loader_val:
        loader_val = t.utils.data.DataLoader(dataset_val, len(dataset_val), shuffle = False, num_workers = 0)
    val_metrics = eval_network_mae_mse_gll(bound_min, bound_max, model, loader_val, loss_fn, len(dataset_val))
    print('Absolute error - validation', val_metrics[ABS_ERR])
    print('R2 - validation', val_metrics[R_SQUARED])

    loader_test = t.utils.data.DataLoader(dataset_test, len(dataset_test), shuffle = False, num_workers = 0)
    test_metrics = eval_network_mae_mse_gll(bound_min, bound_max, model, loader_test, loss_fn, len(dataset_test), is_eval_bounded = False)
    print('Absolute error - test', test_metrics[ABS_ERR])
    print('R2 - test', test_metrics[R_SQUARED])

def plot_and_evaluate_model_gll(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test, root_folder,
                                checkpoint_name, criterion, is_optimized = True, input_size = 1, is_liniar = False, loader_val = None):

    checkpoint = t.load(root_folder + '/' + checkpoint_name + (' best.tar' if is_optimized else '.tar'))
    conf = checkpoint['conf']
    model = get_dense_net(1 if is_liniar else conf['nb_layers'], input_size, conf['layer_size'], conf['dropout_rate'])

    plot_dataset_and_net(checkpoint, model, x_mean, x_std, y_mean, y_std, dataset_val)
    save_fig_in_checkpoint_folder(root_folder, checkpoint_name)

    plot_dataset_and_net(checkpoint, model, x_mean, x_std, y_mean, y_std, dataset_val, with_std=True)
    save_fig_in_checkpoint_folder(root_folder, checkpoint_name, suffix='-with-std')

    if 'sigma' in checkpoint:
        loss_fn = criterion(checkpoint['sigma'])
    elif 'gamma' in checkpoint:
        loss_fn = criterion(checkpoint['gamma'])
    else:
        raise 'sigma or gamma must be provided in checkpoint'

    if not loader_val:
        loader_val = t.utils.data.DataLoader(dataset_val, len(dataset_val), shuffle = False, num_workers = 0)
    val_metrics = eval_network_mae_mse_gll(bound_min, bound_max, model, loader_val, loss_fn, len(dataset_val))
    print('Absolute error - validation', val_metrics[ABS_ERR])
    print('R2 - validation', val_metrics[R_SQUARED])

    loader_test = t.utils.data.DataLoader(dataset_test, len(dataset_test), shuffle = False, num_workers = 0)
    test_metrics = eval_network_mae_mse_gll(bound_min, bound_max, model, loader_test, loss_fn, len(dataset_test), is_eval_bounded = False)
    print('Absolute error - test', test_metrics[ABS_ERR])
    print('R2 - test', test_metrics[R_SQUARED])

def plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test, root_folder, checkpoint_name,
                                            is_optimized = True, input_size = 1, is_liniar = False, truncated_low = None, truncated_high = None):
    censored_collate_fn = distinguish_censored_versus_observed_data(bound_min, bound_max)
    uncensored_collate_fn = distinguish_censored_versus_observed_data(-math.inf, math.inf)
    model = model_fn()
    checkpoint = t.load(root_folder + '/' + checkpoint_name + (' best.tar' if is_optimized else '.tar'))
    if not ('gamma' in checkpoint or 'sigma' in checkpoint):
        raise 'Sigma or gamma must be found in checkpoint'

    plot_dataset_and_net(checkpoint, model, x_mean, x_std, y_mean, y_std, dataset_val)
    save_fig_in_checkpoint_folder(root_folder, checkpoint_name)

    plot_dataset_and_net(checkpoint, model, x_mean, x_std, y_mean, y_std, dataset_val, with_std=True)
    save_fig_in_checkpoint_folder(root_folder, checkpoint_name, suffix='-with-std')

    if 'gamma' in checkpoint:
        loss_fn = Reparametrized_Scaled_Tobit_Loss(checkpoint['gamma'], get_device(), truncated_low = truncated_low, truncated_high = truncated_high)
    elif 'sigma' in checkpoint:
        loss_fn = Scaled_Tobit_Loss(checkpoint['sigma'], get_device(), truncated_low = truncated_low, truncated_high = truncated_high)
    else:
        raise 'sigma or gamma must be provided in checkpoint'

    loader_val = t.utils.data.DataLoader(dataset_val, batch_size = len(dataset_val), shuffle = False, num_workers = 0, collate_fn = censored_collate_fn)
    val_metrics = eval_network_tobit_fixed_std(bound_min, bound_max, model, loader_val, loss_fn, len(dataset_val))
    print('Absolute error - validation', val_metrics[ABS_ERR])
    print('R2 - validation', val_metrics[R_SQUARED])

    loader_test = t.utils.data.DataLoader(dataset_test, batch_size = len(dataset_test), shuffle = False, num_workers = 0, collate_fn = uncensored_collate_fn)
    test_metrics = eval_network_tobit_fixed_std(bound_min, bound_max, model, loader_test, loss_fn, len(dataset_test), is_eval_bounded = False)
    print('Absolute error - test', test_metrics[ABS_ERR])
    print('R2 - test', test_metrics[R_SQUARED])

    if 'gamma' in checkpoint:
        print('\nstd', 1 / checkpoint['gamma'])
    elif 'sigma' in checkpoint:
        print('\nstd', checkpoint['sigma'])


def plot_and_evaluate_model_tobit_dyn_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test, root_folder,
                                          checkpoint_name, is_optimized = True, input_size = 1, is_liniar = False, truncated_low = None, truncated_high = None, is_reparam=False):
    censored_collate_fn = distinguish_censored_versus_observed_data(bound_min, bound_max)
    uncensored_collate_fn = distinguish_censored_versus_observed_data(-math.inf, math.inf)
    model = model_fn()
    checkpoint = t.load(root_folder + '/' + checkpoint_name + (' best.tar' if is_optimized else '.tar'))
    if not ('gamma' in checkpoint or 'sigma' in checkpoint):
        raise 'Sigma or gamma must be found in checkpoint'
    model.load_state_dict(checkpoint['model'])
    model.eval()

    scale_model = get_scale_network()
    scale_model.load_state_dict(checkpoint['gamma' if is_reparam else 'sigma'])
    scale_model.eval()

    plot_dataset_and_net(checkpoint, model, x_mean, x_std, y_mean, y_std, dataset_val, scale_model=scale_model)
    save_fig_in_checkpoint_folder(root_folder, checkpoint_name)

    plot_dataset_and_net(checkpoint, model, x_mean, x_std, y_mean, y_std, dataset_val, scale_model=scale_model, with_std=True)
    save_fig_in_checkpoint_folder(root_folder, checkpoint_name, suffix='-with-std')

    if 'gamma' in checkpoint:
        loss_fn = Heteroscedastic_Reparametrized_Scaled_Tobit_Loss(get_device(), truncated_low = truncated_low, truncated_high = truncated_high)
    elif 'sigma' in checkpoint:
        loss_fn = Heteroscedastic_Scaled_Tobit_Loss(get_device(), truncated_low = truncated_low, truncated_high = truncated_high)

    loader_val = t.utils.data.DataLoader(dataset_val, batch_size = len(dataset_val), shuffle = False, num_workers = 0, collate_fn = censored_collate_fn)
    val_metrics = eval_network_tobit_dyn_std(bound_min, bound_max, model, scale_model, loader_val, loss_fn, len(dataset_val), is_reparam = is_reparam)
    print('Absolute error - validation', val_metrics[ABS_ERR])
    print('R2 - validation', val_metrics[R_SQUARED])

    loader_test = t.utils.data.DataLoader(dataset_test, batch_size = len(dataset_test), shuffle = False, num_workers = 0, collate_fn = uncensored_collate_fn)
    test_metrics = eval_network_tobit_dyn_std(bound_min, bound_max, model, scale_model, loader_test, loss_fn, len(dataset_test), is_eval_bounded = False, is_reparam = is_reparam)
    print('Absolute error - test', test_metrics[ABS_ERR])
    print('R2 - test', test_metrics[R_SQUARED])