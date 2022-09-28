from experiments.synthetic.constants import *
from experiments.util import set_random_seed
from experiments.synthetic.constant_noise.dataset import *
from experiments.synthetic.eval_optimized import plot_and_evaluate_model_tobit_fixed_std
from deep_tobit.util import normalize, distinguish_censored_versus_observed_data
from experiments.synthetic.eval_optimized import plot_dataset_and_net
from experiments.tpe_hyperparam_opt import get_objective_fn_tobit_fixed_std, tpe_opt_hyperparam

"""Constants"""
ROOT_DEEP_TOBIT_SCALED = 'experiments/synthetic/constant_noise/tobit_based/scaled_fixed_std/deep_tobit_cens_NO_trunc'
CHECKPOINT_DEEP_TOBIT_SCALED = 'scaled deep tobit model'

ROOT_DEEP_TOBIT_SCALED_TRUNCATED = 'experiments/synthetic/constant_noise/tobit_based/scaled_fixed_std/deep_tobit_cens_WITH_trunc'
CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED = 'scaled truncated deep tobit model'

ROOT_LINEAR_TOBIT_SCALED = 'experiments/synthetic/constant_noise/tobit_based/scaled_fixed_std/liniar_tobit_cens_NO_trunc'
CHECKPOINT_LINEAR_TOBIT_SCALED = 'scaled linear tobit model'

ROOT_LINEAR_TRUNCATED_TOBIT_SCALED = 'experiments/synthetic/constant_noise/tobit_based/scaled_fixed_std/liniar_tobit_cens_WITH_trunc'
CHECKPOINT_LINEAR_TRUNCATED_TOBIT_SCALED = 'scaled truncated linear tobit model'

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








"""# Scaled Deep Tobit"""

objective_deep_NO_trunc = get_objective_fn_tobit_fixed_std(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_DEEP_TOBIT_SCALED}/{CHECKPOINT_DEEP_TOBIT_SCALED}')

def tpe_opt_deep_NO_trunc():
    return tpe_opt_hyperparam(ROOT_DEEP_TOBIT_SCALED, CHECKPOINT_DEEP_TOBIT_SCALED, objective_deep_NO_trunc)

def eval_deep_NO_trunc():
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                            ROOT_DEEP_TOBIT_SCALED, CHECKPOINT_DEEP_TOBIT_SCALED, is_optimized = True)

def plot_deep_NO_trunc():
    checkpoint = t.load(f'{ROOT_DEEP_TOBIT_SCALED}/{CHECKPOINT_DEEP_TOBIT_SCALED} best.tar')
    plot_dataset_and_net(checkpoint, x_mean, x_std, y_mean, y_std, dataset_val)





"""# Scaled Deep Tobit With Truncation"""

objective_deep_WITH_trunc = get_objective_fn_tobit_fixed_std(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_DEEP_TOBIT_SCALED_TRUNCATED}/{CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED}', truncated_low = zero_normalized)

def tpe_opt_deep_WITH_trunc():
    return tpe_opt_hyperparam(ROOT_DEEP_TOBIT_SCALED_TRUNCATED, CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED, objective_deep_WITH_trunc)

def eval_deep_WITH_trunc():
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                            ROOT_DEEP_TOBIT_SCALED_TRUNCATED, CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED, is_optimized = True)

def plot_deep_WITH_trunc():
    checkpoint = t.load(f'{ROOT_DEEP_TOBIT_SCALED_TRUNCATED}/{CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED} best.tar')
    plot_dataset_and_net(checkpoint, x_mean, x_std, y_mean, y_std, dataset_val)






"""# Scaled Linear Tobit"""

objective_lin_NO_trunc = get_objective_fn_tobit_fixed_std(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_LINEAR_TOBIT_SCALED}/{CHECKPOINT_LINEAR_TOBIT_SCALED}', is_liniar = True, plot = False, log = False)

def tpe_opt_lin_NO_trunc():
    return tpe_opt_hyperparam(ROOT_LINEAR_TOBIT_SCALED, CHECKPOINT_LINEAR_TOBIT_SCALED, objective_lin_NO_trunc)

def eval_lin_NO_trunc():
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                            ROOT_LINEAR_TOBIT_SCALED, CHECKPOINT_LINEAR_TOBIT_SCALED, is_liniar=True, is_optimized = True)

def plot_lin_NO_trunc():
    checkpoint = t.load(f'{ROOT_LINEAR_TOBIT_SCALED}/{CHECKPOINT_LINEAR_TOBIT_SCALED} best.tar')
    plot_dataset_and_net(checkpoint, x_mean, x_std, y_mean, y_std, dataset_val, is_liniar=True)






"""# Scaled Linear Tobit With Truncation"""

objective_lin_WITH_trunc = get_objective_fn_tobit_fixed_std(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_LINEAR_TRUNCATED_TOBIT_SCALED}/{CHECKPOINT_LINEAR_TRUNCATED_TOBIT_SCALED}',
    is_liniar = True, truncated_low = zero_normalized)


def tpe_opt_lin_WITH_trunc():
    return tpe_opt_hyperparam(ROOT_LINEAR_TRUNCATED_TOBIT_SCALED, CHECKPOINT_LINEAR_TRUNCATED_TOBIT_SCALED, objective_lin_WITH_trunc)

def eval_linear_tobit_WITH_trunc():
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                            ROOT_LINEAR_TRUNCATED_TOBIT_SCALED, CHECKPOINT_LINEAR_TRUNCATED_TOBIT_SCALED, is_liniar=True, is_optimized = True)

def plot_linear_tobit_WITH_trunc():
    checkpoint = t.load(f'{ROOT_LINEAR_TRUNCATED_TOBIT_SCALED}/{CHECKPOINT_LINEAR_TRUNCATED_TOBIT_SCALED} best.tar')
    plot_dataset_and_net(checkpoint, x_mean, x_std, y_mean, y_std, dataset_val, is_liniar=True)

# eval_deep_tobit_WITH_trunc()
# eval_deep_tobit_NO_trunc()
# eval_linear_tobit_WITH_trunc()
# eval_linear_tobit_NO_trunc()