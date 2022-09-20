from experiments.constants import GRID_RESULTS_FILE
from experiments.synthetic.constants import *
from experiments.util import set_random_seed
from experiments.synthetic.constant_noise.dataset import *
from experiments.synthetic.eval_optimized import plot_and_evaluate_model_tobit_fixed_std
from deep_tobit.util import normalize, distinguish_censored_versus_observed_data
from experiments.synthetic.models import DenseNetwork
from experiments.synthetic.eval_optimized import plot_dataset_and_net
from experiments.tpe_hyperparam_opt import get_objective_fn_tobit_fixed_std, tpe_opt_hyperparam

"""Constants"""
ROOT_DEEP_TOBIT_REPARAMETRIZED = 'experiments/synthetic/constant_noise/tobit_based/reparam_fixed_std/deep_tobit_cens_NO_trunc'
CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED = 'reparametrized deep tobit model'

ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED = 'experiments/synthetic/constant_noise/tobit_based/reparam_fixed_std/deep_tobit_cens_WITH_trunc'
CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED = 'reparametrized truncated deep tobit model'

ROOT_LINEAR_TOBIT_REPARAMETRIZED = 'experiments/synthetic/constant_noise/tobit_based/reparam_fixed_std/liniar_tobit_NO_trunc'
CHECKPOINT_LINEAR_TOBIT_REPARAMETRIZED = 'reparametrized linear tobit model'

ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED = 'experiments/synthetic/constant_noise/tobit_based/reparam_fixed_std/liniar_tobit_WITH_trunc'
CHECKPOINT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED = 'reparametrized truncated linear tobit model'

"""Reproducible experiments"""

set_random_seed()

"""# Datasets"""

x_mean, x_std, y_mean, y_std = calculate_mean_std(lower_bound = CENSOR_LOW_BOUND, upper_bound = CENSOR_HIGH_BOUND)
print('x mean =', x_mean, 'x std =', x_std, 'y mean =', y_mean, 'y std =', y_std)

dataset_train = TruncatedBetaDistributionDataset(x_mean, x_std, y_mean, y_std, lower_bound = CENSOR_LOW_BOUND, upper_bound = CENSOR_HIGH_BOUND)
dataset_val = TruncatedBetaDistributionValidationDataset(x_mean, x_std, y_mean, y_std, lower_bound = CENSOR_LOW_BOUND, upper_bound = CENSOR_HIGH_BOUND, nb_samples = 1000)
dataset_test = TruncatedBetaDistributionValidationDataset(x_mean, x_std, y_mean, y_std)

"""# Training"""

bound_min = normalize(CENSOR_LOW_BOUND, y_mean, y_std)
bound_max = normalize(CENSOR_HIGH_BOUND, y_mean, y_std)
zero_normalized = normalize(0, y_mean, y_std)

"""# Common Tobit Setup"""

censored_collate_fn = distinguish_censored_versus_observed_data(bound_min, bound_max)
uncensored_collate_fn = distinguish_censored_versus_observed_data(-math.inf, math.inf)

tobit_loader_train = t.utils.data.DataLoader(dataset_train, batch_size = 100, shuffle = True, num_workers = 0, collate_fn = censored_collate_fn)
tobit_loader_val = t.utils.data.DataLoader(dataset_val, batch_size = len(dataset_val), shuffle = False, num_workers = 0, collate_fn = censored_collate_fn)
tobit_loader_test = t.utils.data.DataLoader(dataset_test, batch_size = len(dataset_test), shuffle = False, num_workers = 0, collate_fn = uncensored_collate_fn)




"""# Reparametrized Deep Tobit"""

objective_deep_NO_trunc = get_objective_fn_tobit_fixed_std(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_DEEP_TOBIT_REPARAMETRIZED}/{CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED}', plot = False, log = False, isReparam=True)

def tpe_opt_deep_NO_trunc_reparam():
    return tpe_opt_hyperparam(ROOT_DEEP_TOBIT_REPARAMETRIZED, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED, objective_deep_NO_trunc)

def eval_deep_NO_trunc_reparam():
  plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                          ROOT_DEEP_TOBIT_REPARAMETRIZED, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED, is_optimized= True)

def plot_deep_tobit_NO_trunc_reparam():
    checkpoint = t.load(f'{ROOT_DEEP_TOBIT_REPARAMETRIZED}/{CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED} best.tar')
    plot_dataset_and_net(checkpoint, DenseNetwork(), x_mean, x_std, y_mean, y_std, dataset_val)







"""# Reparametrized Deep Tobit With Truncation"""

objective_deep_WITH_trunc = get_objective_fn_tobit_fixed_std(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED}/{CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED}',
    plot = False, log = False, truncated_low = zero_normalized, isReparam=True)

def tpe_opt_deep_WITH_trunc_reparam():
    return tpe_opt_hyperparam(ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, objective_deep_WITH_trunc)

def eval_deep_WITH_trunc_reparam():
  plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                          ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, is_optimized= True)

def plot_deep_WITH_trunc_reparam():
    checkpoint = t.load(f'{ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED}/{CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED} best.tar')
    plot_dataset_and_net(checkpoint, DenseNetwork(), x_mean, x_std, y_mean, y_std, dataset_val)






"""# Reparametrized Linear Tobit"""

objective_lin_NO_trunc = get_objective_fn_tobit_fixed_std(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_LINEAR_TOBIT_REPARAMETRIZED}/{CHECKPOINT_LINEAR_TOBIT_REPARAMETRIZED}',
    is_liniar = True, plot = False, log = False, isReparam=True)

def tpe_opt_lin_NO_trunc_reparam():
    return tpe_opt_hyperparam(ROOT_LINEAR_TOBIT_REPARAMETRIZED, CHECKPOINT_LINEAR_TOBIT_REPARAMETRIZED, objective_lin_NO_trunc)

def eval_linear_tobit_NO_trunc_reparam():
  plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                          ROOT_LINEAR_TOBIT_REPARAMETRIZED, CHECKPOINT_LINEAR_TOBIT_REPARAMETRIZED, is_liniar=True, is_optimized= True)

def plot_linear_tobit_NO_trunc_reparam():
    checkpoint = t.load(f'{ROOT_LINEAR_TOBIT_REPARAMETRIZED}/{CHECKPOINT_LINEAR_TOBIT_REPARAMETRIZED} best.tar')
    plot_dataset_and_net(checkpoint, t.nn.Linear(1, 1), x_mean, x_std, y_mean, y_std, dataset_val)







"""# Reparametrized Linear Tobit With Truncation"""

objective_lin_WITH_trunc = get_objective_fn_tobit_fixed_std(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED}/{CHECKPOINT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED}',
    is_liniar = True, plot = False, log = False, truncated_low = zero_normalized, isReparam=True)

def tpe_opt_lin_WITH_trunc_reparam():
    return tpe_opt_hyperparam(ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED, CHECKPOINT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED, objective_lin_WITH_trunc)

def eval_linear_tobit_WITH_trunc_reparam():
  plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                          ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED, CHECKPOINT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED, is_liniar=True, is_optimized= True)

def plot_linear_tobit_WITH_trunc_reparam():
    checkpoint = t.load(f'{ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED}/{CHECKPOINT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED} best.tar')
    plot_dataset_and_net(checkpoint, t.nn.Linear(1, 1), x_mean, x_std, y_mean, y_std, dataset_val)
