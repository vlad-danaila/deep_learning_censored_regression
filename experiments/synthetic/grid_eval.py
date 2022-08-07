import math
import matplotlib.pyplot as plt
import torch as t
from deep_tobit.loss import Reparametrized_Scaled_Tobit_Loss, Scaled_Tobit_Loss, \
    Heteroscedastic_Reparametrized_Scaled_Tobit_Loss, Heteroscedastic_Scaled_Tobit_Loss
from deep_tobit.util import distinguish_censored_versus_observed_data

from experiments.constants import ABS_ERR, R_SQUARED
from experiments.synthetic.models import DenseNetwork, get_scale_network
from experiments.synthetic.plot import plot_beta, plot_dataset, plot_net, plot_fixed_and_dynamic_std
from experiments.train import eval_network_mae_mse_gll, eval_network_tobit_fixed_std, eval_network_tobit_dyn_std

def plot_dataset_and_net(checkpoint, model, x_mean, x_std, y_mean, y_std, dataset_val, with_std = False):
    model.load_state_dict(checkpoint['model'])
    plot_beta(x_mean, x_std, y_mean, y_std, label = 'true distribution')
    # plot_dataset(dataset_test, size = .3, label = 'test data')
    plot_dataset(dataset_val, label = 'validation data')
    if 'sigma' in checkpoint:
        plot_net(model, dataset_val, sigma = checkpoint['sigma'], with_std = with_std)
    elif 'gamma' in checkpoint:
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
                                    checkpoint_name, criterion, isGrid = True, model_fn = DenseNetwork, is_gamma = False, loader_val = None):
    model = model_fn()
    loss_fn = criterion()
    checkpoint = t.load(root_folder + '/' + ('grid ' if isGrid else '') + checkpoint_name + '.tar')
    plot_dataset_and_net(checkpoint, model, x_mean, x_std, y_mean, y_std, dataset_val)
    plt.savefig('{}.pdf'.format(root_folder + '/' + checkpoint_name), dpi = 300, format = 'pdf')
    plt.savefig('{}.svg'.format(root_folder + '/' + checkpoint_name), dpi = 300, format = 'svg')
    plt.savefig('{}.png'.format(root_folder + '/' + checkpoint_name), dpi = 200, format = 'png')
    plt.close()

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
                                checkpoint_name, criterion, isGrid = True, model_fn = DenseNetwork, loader_val = None):
    model = model_fn()
    checkpoint = t.load(root_folder + '/' + ('grid ' if isGrid else '') + checkpoint_name + '.tar')
    plot_dataset_and_net(checkpoint, model, x_mean, x_std, y_mean, y_std, dataset_val)
    plt.savefig('{}.pdf'.format(root_folder + '/' + checkpoint_name), dpi = 300, format = 'pdf')
    plt.savefig('{}.svg'.format(root_folder + '/' + checkpoint_name), dpi = 300, format = 'svg')
    plt.savefig('{}.png'.format(root_folder + '/' + checkpoint_name), dpi = 200, format = 'png')
    plt.close()

    plot_dataset_and_net(checkpoint, model, x_mean, x_std, y_mean, y_std, dataset_val, with_std=True)
    plt.savefig('{}-with-std.pdf'.format(root_folder + '/' + checkpoint_name), dpi = 300, format = 'pdf')
    plt.savefig('{}-with-std.svg'.format(root_folder + '/' + checkpoint_name), dpi = 300, format = 'svg')
    plt.savefig('{}-with-std.png'.format(root_folder + '/' + checkpoint_name), dpi = 200, format = 'png')
    plt.close()

    if 'sigma' in checkpoint:
        loss_fn = criterion(checkpoint['sigma'])
    elif 'gamma' in checkpoint:
        loss_fn = criterion(checkpoint['gamma'])

    if not loader_val:
        loader_val = t.utils.data.DataLoader(dataset_val, len(dataset_val), shuffle = False, num_workers = 0)
    val_metrics = eval_network_mae_mse_gll(bound_min, bound_max, model, loader_val, loss_fn, len(dataset_val))
    print('Absolute error - validation', val_metrics[ABS_ERR])
    print('R2 - validation', val_metrics[R_SQUARED])

    loader_test = t.utils.data.DataLoader(dataset_test, len(dataset_test), shuffle = False, num_workers = 0)
    test_metrics = eval_network_mae_mse_gll(bound_min, bound_max, model, loader_test, loss_fn, len(dataset_test), is_eval_bounded = False)
    print('Absolute error - test', test_metrics[ABS_ERR])
    print('R2 - test', test_metrics[R_SQUARED])


def plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test, root_folder, checkpoint_name, isGrid = True, model_fn = DenseNetwork, truncated_low = None, truncated_high = None):
    censored_collate_fn = distinguish_censored_versus_observed_data(bound_min, bound_max)
    uncensored_collate_fn = distinguish_censored_versus_observed_data(-math.inf, math.inf)
    model = model_fn()
    checkpoint = t.load(root_folder + '/' + ('grid ' if isGrid else '') + checkpoint_name + '.tar')
    if not ('gamma' in checkpoint or 'sigma' in checkpoint):
        raise 'Sigma or gamma must be found in checkpoint'
    model.load_state_dict(checkpoint['model'])
    plot_beta(x_mean, x_std, y_mean, y_std, label = 'true distribution')
    plot_dataset(dataset_val, label = 'validation data')
    if 'gamma' in checkpoint:
        plot_net(model, dataset_val, gamma = checkpoint['gamma'])
    elif 'sigma' in checkpoint:
        plot_net(model, dataset_val, sigma = checkpoint['sigma'])
    plt.xlabel('input (standardized)')
    plt.ylabel('outcome (standardized)')
    plt.ylim((-2.5, 2.5))
    lgnd = plt.legend()
    lgnd.legendHandles[0]._sizes = [10]
    lgnd.legendHandles[1]._sizes = [10]
    lgnd.legendHandles[2]._sizes = [10]
    plt.savefig('{}.pdf'.format(root_folder + '/' + checkpoint_name), dpi = 300, format = 'pdf')
    plt.savefig('{}.svg'.format(root_folder + '/' + checkpoint_name), dpi = 300, format = 'svg')
    plt.savefig('{}.png'.format(root_folder + '/' + checkpoint_name), dpi = 200, format = 'png')
    plt.close()

    plot_beta(x_mean, x_std, y_mean, y_std, label = 'true distribution')
    plot_dataset(dataset_val, label = 'validation data')
    if 'gamma' in checkpoint:
        plot_net(model, dataset_val, gamma = checkpoint['gamma'], with_std = True)
    elif 'sigma' in checkpoint:
        plot_net(model, dataset_val, sigma = checkpoint['sigma'], with_std = True)
    plt.xlabel('input (standardized)')
    plt.ylabel('outcome (standardized)')
    plt.ylim((-2.5, 2.5))
    lgnd = plt.legend()
    lgnd.legendHandles[0]._sizes = [10]
    lgnd.legendHandles[1]._sizes = [10]
    lgnd.legendHandles[2]._sizes = [10]
    plt.savefig('{}-with-std.pdf'.format(root_folder + '/' + checkpoint_name), dpi = 300, format = 'pdf')
    plt.savefig('{}-with-std.svg'.format(root_folder + '/' + checkpoint_name), dpi = 300, format = 'svg')
    plt.savefig('{}-with-std.png'.format(root_folder + '/' + checkpoint_name), dpi = 200, format = 'png')
    plt.close()

    if 'gamma' in checkpoint:
        loss_fn = Reparametrized_Scaled_Tobit_Loss(checkpoint['gamma'], 'cpu', truncated_low = truncated_low, truncated_high = truncated_high)
    elif 'sigma' in checkpoint:
        loss_fn = Scaled_Tobit_Loss(checkpoint['sigma'], 'cpu', truncated_low = truncated_low, truncated_high = truncated_high)
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
                                          checkpoint_name, isGrid = True, model_fn = DenseNetwork, truncated_low = None, truncated_high = None, is_reparam=False):
    censored_collate_fn = distinguish_censored_versus_observed_data(bound_min, bound_max)
    uncensored_collate_fn = distinguish_censored_versus_observed_data(-math.inf, math.inf)
    model = model_fn()
    checkpoint = t.load(root_folder + '/' + ('grid ' if isGrid else '') + checkpoint_name + '.tar')
    if not ('gamma' in checkpoint or 'sigma' in checkpoint):
        raise 'Sigma or gamma must be found in checkpoint'
    model.load_state_dict(checkpoint['model'])
    model.eval()

    scale_model = get_scale_network()
    if is_reparam:
        scale_model.load_state_dict(checkpoint['gamma'])
    else:
        scale_model.load_state_dict(checkpoint['sigma'])
    scale_model.eval()

    plot_beta(x_mean, x_std, y_mean, y_std, label = 'true distribution')
    plot_dataset(dataset_val, label = 'validation data')
    if 'gamma' in checkpoint:
        plot_net(model, dataset_val, gamma_model = scale_model)
    elif 'sigma' in checkpoint:
        plot_net(model, dataset_val, sigma_model = scale_model)
    plt.xlabel('input (standardized)')
    plt.ylabel('outcome (standardized)')
    plt.ylim((-2.5, 2.5))
    lgnd = plt.legend()
    lgnd.legendHandles[0]._sizes = [10]
    lgnd.legendHandles[1]._sizes = [10]
    lgnd.legendHandles[2]._sizes = [10]
    plt.savefig('{}.pdf'.format(root_folder + '/' + checkpoint_name), dpi = 300, format = 'pdf')
    plt.savefig('{}.svg'.format(root_folder + '/' + checkpoint_name), dpi = 300, format = 'svg')
    plt.savefig('{}.png'.format(root_folder + '/' + checkpoint_name), dpi = 200, format = 'png')
    plt.close()

    plot_beta(x_mean, x_std, y_mean, y_std, label = 'true distribution')
    plot_dataset(dataset_val, label = 'validation data')
    if 'gamma' in checkpoint:
        plot_net(model, dataset_val, gamma_model = scale_model, with_std = True)
    elif 'sigma' in checkpoint:
        plot_net(model, dataset_val, sigma_model = scale_model, with_std = True)
    plt.xlabel('input (standardized)')
    plt.ylabel('outcome (standardized)')
    plt.ylim((-2.5, 2.5))
    lgnd = plt.legend()
    lgnd.legendHandles[0]._sizes = [10]
    lgnd.legendHandles[1]._sizes = [10]
    lgnd.legendHandles[2]._sizes = [10]
    lgnd.legendHandles[3]._sizes = [10]
    plt.savefig('{}-with-std.pdf'.format(root_folder + '/' + checkpoint_name), dpi = 300, format = 'pdf')
    plt.savefig('{}-with-std.svg'.format(root_folder + '/' + checkpoint_name), dpi = 300, format = 'svg')
    plt.savefig('{}-with-std.png'.format(root_folder + '/' + checkpoint_name), dpi = 200, format = 'png')
    plt.close()

    # TODO put real fixed std as a parameter not hardcoded
    plot_fixed_and_dynamic_std(dataset_val, model, scale_model, 0.5203 if is_reparam else 0.4017, is_reparam=is_reparam)
    plt.xlabel('unidimensional PCA')
    plt.ylabel('standard deviation')
    lgnd = plt.legend()
    lgnd.legendHandles[0]._sizes = [10]
    lgnd.legendHandles[1]._sizes = [10]
    plt.savefig('{}-two-std.pdf'.format(checkpoint_name), dpi = 300, format = 'pdf')
    plt.savefig('{}-two-std.svg'.format(checkpoint_name), dpi = 300, format = 'svg')
    plt.savefig('{}-two-std.png'.format(checkpoint_name), dpi = 200, format = 'png')
    plt.close()

    if 'gamma' in checkpoint:
        loss_fn = Heteroscedastic_Reparametrized_Scaled_Tobit_Loss('cpu', truncated_low = truncated_low, truncated_high = truncated_high)
    elif 'sigma' in checkpoint:
        loss_fn = Heteroscedastic_Scaled_Tobit_Loss('cpu', truncated_low = truncated_low, truncated_high = truncated_high)
    loader_val = t.utils.data.DataLoader(dataset_val, batch_size = len(dataset_val), shuffle = False, num_workers = 0, collate_fn = censored_collate_fn)
    val_metrics = eval_network_tobit_dyn_std(bound_min, bound_max, model, scale_model, loader_val, loss_fn, len(dataset_val), is_reparam = is_reparam)
    print('Absolute error - validation', val_metrics[ABS_ERR])
    print('R2 - validation', val_metrics[R_SQUARED])

    loader_test = t.utils.data.DataLoader(dataset_test, batch_size = len(dataset_test), shuffle = False, num_workers = 0, collate_fn = uncensored_collate_fn)
    test_metrics = eval_network_tobit_dyn_std(bound_min, bound_max, model, scale_model, loader_test, loss_fn, len(dataset_test), is_eval_bounded = False, is_reparam = is_reparam)
    print('Absolute error - test', test_metrics[ABS_ERR])
    print('R2 - test', test_metrics[R_SQUARED])