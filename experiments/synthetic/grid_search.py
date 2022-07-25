from sklearn.model_selection import ParameterGrid
import os
from experiments.synthetic.constants import *
from experiments.models import DenseNetwork
from experiments.synthetic.constant_noise.dataset import *
from experiments.synthetic.train import train_network_mae_mse_gll, eval_network_mae_mse_gll, train_network_tobit, eval_network_tobit
from experiments.synthetic.plot import *
from deep_tobit.util import distinguish_censored_versus_observed_data
from deep_tobit.loss import Scaled_Tobit_Loss

def get_grid_search_space():
    return [{
        'max_lr': [1e-5],
        'epochs': [10],
        'batch': [ 200],
        'pct_start': [0.45],
        'anneal_strategy': ['linear'],
        'base_momentum': [0.85],
        'max_momentum': [0.95],
        'div_factor': [10, 5],
        'final_div_factor': [1e4],
        'weight_decay': [0]
    }]
    # return [{
    #     'max_lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
    #     'epochs': [10, 20],
    #     'batch': [100, 200],
    #     'pct_start': [0.45],
    #     'anneal_strategy': ['linear'],
    #     'base_momentum': [0.85],
    #     'max_momentum': [0.95],
    #     'div_factor': [10, 5, 2],
    #     'final_div_factor': [1e4],
    #     'weight_decay': [0]
    # }]

"""# Grid Search Setup"""

def grid_search(root_folder, dataset_train, dataset_val, bound_min, bound_max, grid_config, train_callback, checkpoint_name, nb_iterations = 1, conf_validation = None):
    configs = ParameterGrid(grid_config)
    configs_len = len(configs)
    counter = 0
    checkpoint_file = root_folder + '/' + checkpoint_name + '.tar'
    grid_checkpoint_file = root_folder + '/grid ' + checkpoint_name + '.tar'
    try:
        resume_grid_search = t.load(root_folder + '/' + GRID_RESULTS_FILE)
    except FileNotFoundError:
        resume_grid_search = None

    results = {}
    best = [math.inf, math.inf, -math.inf]
    if resume_grid_search is not None and 'best' in resume_grid_search:
        best_conf = resume_grid_search['best']
        print('Best previous configuration', best_conf)
        best = resume_grid_search[str(best_conf)]
        print(f'Best previous metrics abs err = {best[ABS_ERR]}, R2 = {best[R_SQUARED]}')
        results = resume_grid_search

    for conf in ParameterGrid(grid_config):
        counter += 1

        if resume_grid_search is not None and str(conf) in resume_grid_search:
            print('Allready evaluated configuration', conf)
            continue

        if not conf_validation(conf):
            print('Skipping over configuration', conf)
            results[str(conf)] = 'invalid'
            continue

        print('-' * 5, 'grid search {}/{}'.format(counter, configs_len), '-' * 5)
        print('Config:', conf)

        best_from_iterations = [math.inf, math.inf, -math.inf]

        for i in range(nb_iterations):
            if nb_iterations != 1:
                print('Iteration', i + 1)
            metrics = train_callback(dataset_train, dataset_val, bound_min, bound_max, conf)

            # if metrics[R_SQUARED] > best[R_SQUARED]:
            if metrics[ABS_ERR] < best[ABS_ERR]:
                best_from_iterations = metrics

            # if metrics[R_SQUARED] > best[R_SQUARED]:
            if metrics[ABS_ERR] < best[ABS_ERR] and not (math.isnan(metrics[LOSS] or math.isnan(metrics[ABS_ERR]) or math.isnan(metrics[R_SQUARED]))):
                best = metrics
                results['best'] = conf
                if os.path.exists(grid_checkpoint_file):
                    os.remove(grid_checkpoint_file)
                os.rename(checkpoint_file, grid_checkpoint_file)
        else:
            results[str(conf)] = best_from_iterations
            t.save(results, root_folder + '/' + GRID_RESULTS_FILE)

    return best

def train_and_evaluate_mae_mse(checkpoint, criterion, model_fn = DenseNetwork, plot = False, log = True, is_gamma = False):
    def grid_callback(dataset_train, dataset_val, bound_min, bound_max, conf):
        model = model_fn()
        loader_train = t.utils.data.DataLoader(dataset_train, conf['batch'], shuffle = False, num_workers = 0)
        loader_val = t.utils.data.DataLoader(dataset_val, len(dataset_val), shuffle = False, num_workers = 0)
        loss_fn = criterion()
        params = model.parameters()
        optimizer = t.optim.SGD(params, lr = conf['max_lr'] / conf['div_factor'], momentum = conf['max_momentum'], weight_decay = conf['weight_decay'])
        scheduler = t.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr = conf['max_lr'],
            steps_per_epoch = len(loader_train),
            epochs = conf['epochs'],
            pct_start = conf['pct_start'],
            anneal_strategy = conf['anneal_strategy'],
            base_momentum = conf['base_momentum'],
            max_momentum = conf['max_momentum'],
            div_factor = conf['div_factor'],
            final_div_factor = conf['final_div_factor']
        )
        train_metrics, val_metrics, best = train_network_mae_mse_gll(bound_min, bound_max,
                                                                     model, loss_fn, optimizer, scheduler, loader_train, loader_val, checkpoint, conf['batch'], len(dataset_val), conf['epochs'], log = log)
        if plot:
            plot_epochs(train_metrics, val_metrics)
        return best
    return grid_callback

def train_and_evaluate_gll(checkpoint, criterion, model_fn = DenseNetwork, plot = False, log = True):
    def grid_callback(dataset_train, dataset_val, bound_min, bound_max, conf):
        model = model_fn()
        loader_train = t.utils.data.DataLoader(dataset_train, conf['batch'], shuffle = False, num_workers = 0)
        loader_val = t.utils.data.DataLoader(dataset_val, len(dataset_val), shuffle = False, num_workers = 0)
        scale = t.tensor(1., requires_grad = True) # could be sigma or gamma
        loss_fn = criterion(scale)
        params = [
            {'params': model.parameters()},
            {'params': scale}
        ]
        optimizer = t.optim.SGD(params, lr = conf['max_lr'] / conf['div_factor'], momentum = conf['max_momentum'], weight_decay = conf['weight_decay'])
        scheduler = t.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr = conf['max_lr'],
            steps_per_epoch = len(loader_train),
            epochs = conf['epochs'],
            pct_start = conf['pct_start'],
            anneal_strategy = conf['anneal_strategy'],
            base_momentum = conf['base_momentum'],
            max_momentum = conf['max_momentum'],
            div_factor = conf['div_factor'],
            final_div_factor = conf['final_div_factor']
        )
        train_metrics, val_metrics, best = train_network_mae_mse_gll(bound_min, bound_max,
              model, loss_fn, optimizer, scheduler, loader_train, loader_val, checkpoint, conf['batch'], len(dataset_val), conf['epochs'], log = log)
        if plot:
            plot_epochs(train_metrics, val_metrics)
        return best
    return grid_callback

def train_and_evaluate_tobit(checkpoint, model_fn = DenseNetwork, plot = False, log = True, device = 'cpu', truncated_low = None, truncated_high = None):
    def grid_callback(dataset_train, dataset_val, bound_min, bound_max, conf):
        censored_collate_fn = distinguish_censored_versus_observed_data(bound_min, bound_max)
        model = model_fn()
        loader_train = t.utils.data.DataLoader(dataset_train, batch_size = conf['batch'], shuffle = True, num_workers = 0, collate_fn = censored_collate_fn)
        loader_val = t.utils.data.DataLoader(dataset_val, batch_size = len(dataset_val), shuffle = False, num_workers = 0, collate_fn = censored_collate_fn)
        sigma = t.tensor(1., requires_grad = True)
        loss_fn = Scaled_Tobit_Loss(sigma, device, truncated_low = truncated_low, truncated_high = truncated_high)
        params = [
            {'params': model.parameters()},
            {'params': sigma}
        ]
        optimizer = t.optim.SGD(params, lr = conf['max_lr'] / conf['div_factor'], momentum = conf['max_momentum'], weight_decay = conf['weight_decay'])
        scheduler = t.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr = conf['max_lr'],
            steps_per_epoch = len(loader_train),
            epochs = conf['epochs'],
            pct_start = conf['pct_start'],
            anneal_strategy = conf['anneal_strategy'],
            base_momentum = conf['base_momentum'],
            max_momentum = conf['max_momentum'],
            div_factor = conf['div_factor'],
            final_div_factor = conf['final_div_factor']
        )
        train_metrics, val_metrics, best = train_network_tobit(bound_min, bound_max,
            model, loss_fn, optimizer, scheduler, loader_train, loader_val, checkpoint, conf['batch'], len(dataset_val), conf['epochs'], log = log)
        if plot:
            plot_epochs(train_metrics, val_metrics)
        return best
    return grid_callback

def config_validation(conf):
    return conf['div_factor'] <= conf['final_div_factor'] and conf['max_momentum'] >= conf['base_momentum']

"""# Plot Selected(With Grid) Model"""

def plot_and_evaluate_model_mae_mse(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test, root_folder, checkpoint_name, criterion, isGrid = True, model_fn = DenseNetwork, is_gamma = False, loader_val = None):
    model = model_fn()
    checkpoint = t.load(root_folder + '/' + ('grid ' if isGrid else '') + checkpoint_name + '.tar')
    model.load_state_dict(checkpoint['model'])
    plot_beta(x_mean, x_std, y_mean, y_std, label = 'true distribution')
    # plot_dataset(dataset_test, size = .3, label = 'test data')
    plot_dataset(dataset_val, size = .3, label = 'validation data')
    plot_net(model, dataset_val)
    loss_fn = criterion()
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

    if not loader_val:
        loader_val = t.utils.data.DataLoader(dataset_val, len(dataset_val), shuffle = False, num_workers = 0)
    val_metrics = eval_network_mae_mse_gll(bound_min, bound_max, model, loader_val, loss_fn, len(dataset_val))
    print('Absolute error - validation', val_metrics[ABS_ERR])
    print('R2 - validation', val_metrics[R_SQUARED])

    loader_test = t.utils.data.DataLoader(dataset_test, len(dataset_test), shuffle = False, num_workers = 0)
    test_metrics = eval_network_mae_mse_gll(bound_min, bound_max, model, loader_test, loss_fn, len(dataset_test), is_eval_bounded = False)
    print('Absolute error - test', test_metrics[ABS_ERR])
    print('R2 - test', test_metrics[R_SQUARED])

def real_y_std():
    real_x_mean, real_x_std, real_y_mean, real_y_std = calculate_mean_std()
    return real_y_std

def plot_and_evaluate_model_gll(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test, root_folder, checkpoint_name, criterion, isGrid = True, model_fn = DenseNetwork, loader_val = None):
    model = model_fn()
    checkpoint = t.load(root_folder + '/' + ('grid ' if isGrid else '') + checkpoint_name + '.tar')
    model.load_state_dict(checkpoint['model'])
    plot_beta(x_mean, x_std, y_mean, y_std, label = 'true distribution')
    # plot_dataset(dataset_test, size = .3, label = 'test data')
    plot_dataset(dataset_val, size = .3, label = 'validation data')
    if 'sigma' in checkpoint:
        plot_net(model, dataset_val, sigma = checkpoint['sigma'])
        loss_fn = criterion(checkpoint['sigma'])
    elif 'gamma' in checkpoint:
        plot_net(model, dataset_val, gamma = checkpoint['gamma'])
        loss_fn = criterion(checkpoint['gamma'])
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
    # plot_dataset(dataset_test, size = .3, label = 'test data')
    plot_dataset(dataset_val, size = .3, label = 'validation data')
    if 'sigma' in checkpoint:
        plot_net(model, dataset_val, sigma = checkpoint['sigma'], with_std = True)
        loss_fn = criterion(checkpoint['sigma'])
    elif 'gamma' in checkpoint:
        plot_net(model, dataset_val, gamma = checkpoint['gamma'], with_std = True)
        loss_fn = criterion(checkpoint['gamma'])
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

    if not loader_val:
        loader_val = t.utils.data.DataLoader(dataset_val, len(dataset_val), shuffle = False, num_workers = 0)
    val_metrics = eval_network_mae_mse_gll(bound_min, bound_max, model, loader_val, loss_fn, len(dataset_val))
    print('Absolute error - validation', val_metrics[ABS_ERR])
    print('R2 - validation', val_metrics[R_SQUARED])

    loader_test = t.utils.data.DataLoader(dataset_test, len(dataset_test), shuffle = False, num_workers = 0)
    test_metrics = eval_network_mae_mse_gll(bound_min, bound_max, model, loader_test, loss_fn, len(dataset_test), is_eval_bounded = False)
    print('Absolute error - test', test_metrics[ABS_ERR])
    print('R2 - test', test_metrics[R_SQUARED])

def plot_and_evaluate_model_tobit(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test, root_folder, checkpoint_name, isGrid = True, model_fn = DenseNetwork, truncated_low = None, truncated_high = None):
    censored_collate_fn = distinguish_censored_versus_observed_data(bound_min, bound_max)
    uncensored_collate_fn = distinguish_censored_versus_observed_data(-math.inf, math.inf)
    model = model_fn()
    checkpoint = t.load(root_folder + '/' + ('grid ' if isGrid else '') + checkpoint_name + '.tar')
    model.load_state_dict(checkpoint['model'])
    plot_beta(x_mean, x_std, y_mean, y_std, label = 'true distribution')
    plot_dataset(dataset_val, size = .3, label = 'validation data')
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
    plot_dataset(dataset_val, size = .3, label = 'validation data')
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

    loss_fn = Scaled_Tobit_Loss(checkpoint['sigma'], 'cpu', truncated_low = truncated_low, truncated_high = truncated_high)

    loader_val = t.utils.data.DataLoader(dataset_val, batch_size = len(dataset_val), shuffle = False, num_workers = 0, collate_fn = censored_collate_fn)
    val_metrics = eval_network_tobit(bound_min, bound_max, model, loader_val, loss_fn, len(dataset_val))
    print('Absolute error - validation', val_metrics[ABS_ERR])
    print('R2 - validation', val_metrics[R_SQUARED])

    loader_test = t.utils.data.DataLoader(dataset_test, batch_size = len(dataset_test), shuffle = False, num_workers = 0, collate_fn = uncensored_collate_fn)
    test_metrics = eval_network_tobit(bound_min, bound_max, model, loader_test, loss_fn, len(dataset_test), is_eval_bounded = False)
    print('Absolute error - test', test_metrics[ABS_ERR])
    print('R2 - test', test_metrics[R_SQUARED])

    print('\nstd', checkpoint['sigma'])