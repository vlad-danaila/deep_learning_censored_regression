import torch as t

from deep_tobit.loss import Reparametrized_Scaled_Tobit_Loss, Scaled_Tobit_Loss, \
    Heteroscedastic_Reparametrized_Scaled_Tobit_Loss, Heteroscedastic_Scaled_Tobit_Loss
from deep_tobit.util import distinguish_censored_versus_observed_data
from experiments.synthetic.models import DenseNetwork, get_scale_network
from experiments.synthetic.plot import plot_epochs
from experiments.train import train_network_mae_mse_gll, train_network_tobit_fixed_std, train_network_tobit_dyn_std
from experiments.train import n, k

def train_and_evaluate_mae_mse(checkpoint, criterion, model_fn = DenseNetwork, plot = False, log = True, is_gamma = False, n=n, k=k):
    def grid_callback(dataset_train, dataset_val, bound_min, bound_max, conf):
        model = model_fn()
        loader_train = t.utils.data.DataLoader(dataset_train, conf['batch'], shuffle = True, num_workers = 0)
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


def train_and_evaluate_gll(checkpoint, criterion, model_fn = DenseNetwork, plot = False, log = True, n=n, k=k):
    def grid_callback(dataset_train, dataset_val, bound_min, bound_max, conf):
        model = model_fn()
        loader_train = t.utils.data.DataLoader(dataset_train, conf['batch'], shuffle = True, num_workers = 0)
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
              model, loss_fn, optimizer, scheduler, loader_train, loader_val, checkpoint, conf['batch'], len(dataset_val), conf['epochs'], log = log, n=n, k=k)
        if plot:
            plot_epochs(train_metrics, val_metrics)
        return best
    return grid_callback


def train_and_evaluate_tobit_fixed_std(checkpoint, model_fn = DenseNetwork, plot = False, log = True, device ='cpu', truncated_low = None, truncated_high = None, isReparam = False, n=n, k=k):
    def grid_callback(dataset_train, dataset_val, bound_min, bound_max, conf):
        censored_collate_fn = distinguish_censored_versus_observed_data(bound_min, bound_max)
        model = model_fn()
        loader_train = t.utils.data.DataLoader(dataset_train, batch_size = conf['batch'], shuffle = True, num_workers = 0, collate_fn = censored_collate_fn)
        loader_val = t.utils.data.DataLoader(dataset_val, batch_size = len(dataset_val), shuffle = False, num_workers = 0, collate_fn = censored_collate_fn)
        scale = t.tensor(1., requires_grad = True)
        if isReparam:
            loss_fn = Reparametrized_Scaled_Tobit_Loss(scale, device, truncated_low = truncated_low, truncated_high = truncated_high)
        else:
            loss_fn = Scaled_Tobit_Loss(scale, device, truncated_low = truncated_low, truncated_high = truncated_high)
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
        train_metrics, val_metrics, best = train_network_tobit_fixed_std(bound_min, bound_max,
            model, loss_fn, optimizer, scheduler, loader_train, loader_val, checkpoint, conf['batch'], len(dataset_val), conf['epochs'], log = log, n=n, k=k)
        if plot:
            plot_epochs(train_metrics, val_metrics)
        return best
    return grid_callback


def train_and_evaluate_tobit_dyn_std(checkpoint, model_fn = DenseNetwork, plot = False, log = True, device = 'cpu',
                truncated_low = None, truncated_high = None, is_reparam = False, n=n, k=k):
    def grid_callback(dataset_train, dataset_val, bound_min, bound_max, conf):
        censored_collate_fn = distinguish_censored_versus_observed_data(bound_min, bound_max)
        model = model_fn()
        loader_train = t.utils.data.DataLoader(dataset_train, batch_size = conf['batch'], shuffle = True, num_workers = 0, collate_fn = censored_collate_fn)
        loader_val = t.utils.data.DataLoader(dataset_val, batch_size = len(dataset_val), shuffle = False, num_workers = 0, collate_fn = censored_collate_fn)
        scale_model = get_scale_network()
        if is_reparam:
            loss_fn = Heteroscedastic_Reparametrized_Scaled_Tobit_Loss(device, truncated_low = truncated_low, truncated_high = truncated_high)
        else:
            loss_fn = Heteroscedastic_Scaled_Tobit_Loss(device, truncated_low = truncated_low, truncated_high = truncated_high)
        params = [
            {'params': model.parameters()},
            {'params': scale_model.parameters()}
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
        train_metrics, val_metrics, best = train_network_tobit_dyn_std(bound_min, bound_max,
              model, scale_model, loss_fn, optimizer, scheduler, loader_train, loader_val, checkpoint, conf['batch'],
                    len(dataset_val), conf['epochs'], grad_clip = conf['grad_clip'], log = log, is_reparam=is_reparam, n=n, k=k)
        if plot:
            plot_epochs(train_metrics, val_metrics)
        return best
    return grid_callback