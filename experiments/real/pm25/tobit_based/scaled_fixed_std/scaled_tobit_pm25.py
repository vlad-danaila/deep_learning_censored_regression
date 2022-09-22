from experiments.util import set_random_seed, load_checkpoint
from experiments.real.pm25.dataset import *
from experiments.real.pm25.eval_optimized import plot_and_evaluate_model_tobit_fixed_std, plot_dataset_and_net
from experiments.tpe_hyperparam_opt import get_objective_fn_tobit_fixed_std, tpe_opt_hyperparam

"""Constants"""
ROOT_DEEP_TOBIT_SCALED = 'experiments/real/pm25/tobit_based/scaled_fixed_std/deep_tobit_cens_NO_trunc'
CHECKPOINT_DEEP_TOBIT_SCALED = 'scaled deep tobit model'

ROOT_DEEP_TOBIT_SCALED_TRUNCATED = 'experiments/real/pm25/tobit_based/scaled_fixed_std/deep_tobit_cens_WITH_trunc'
CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED = 'scaled truncated deep tobit model'

ROOT_LINEAR_TOBIT_SCALED = 'experiments/real/pm25/tobit_based/scaled_fixed_std/liniar_tobit_cens_NO_trunc'
CHECKPOINT_LINEAR_TOBIT_SCALED = 'scaled linear tobit model'

ROOT_LINEAR_TRUNCATED_TOBIT_SCALED = 'experiments/real/pm25/tobit_based/scaled_fixed_std/liniar_tobit_cens_WITH_trunc'
CHECKPOINT_LINEAR_TRUNCATED_TOBIT_SCALED = 'scaled truncated linear tobit model'

"""Reproducible experiments"""

set_random_seed()




"""# Scaled Deep Tobit"""

objective_deep_NO_trunc = get_objective_fn_tobit_fixed_std(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_DEEP_TOBIT_SCALED}/{CHECKPOINT_DEEP_TOBIT_SCALED}', input_size = INPUT_SIZE)

def tpe_opt_deep_NO_trunc():
    return tpe_opt_hyperparam(ROOT_DEEP_TOBIT_SCALED, CHECKPOINT_DEEP_TOBIT_SCALED, objective_deep_NO_trunc)

def eval_deep_NO_trunc():
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                            ROOT_DEEP_TOBIT_SCALED, CHECKPOINT_DEEP_TOBIT_SCALED, is_optimized = True)

def plot_deep_NO_trunc():
    checkpoint = load_checkpoint(f'{ROOT_DEEP_TOBIT_SCALED}/{CHECKPOINT_DEEP_TOBIT_SCALED} best.tar')
    plot_dataset_and_net(checkpoint, test_df(df))





"""# Scaled Deep Tobit With Truncation"""

objective_deep_WITH_trunc = get_objective_fn_tobit_fixed_std(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_DEEP_TOBIT_SCALED_TRUNCATED}/{CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED}', input_size = INPUT_SIZE, truncated_low = zero_normalized)

def tpe_opt_deep_WITH_trunc():
    return tpe_opt_hyperparam(ROOT_DEEP_TOBIT_SCALED_TRUNCATED, CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED, objective_deep_WITH_trunc)

def eval_deep_WITH_trunc():
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                            ROOT_DEEP_TOBIT_SCALED_TRUNCATED, CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED, is_optimized = True)

def plot_deep_WITH_trunc():
    checkpoint = load_checkpoint(f'{ROOT_DEEP_TOBIT_SCALED_TRUNCATED}/{CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED} best.tar')
    plot_dataset_and_net(checkpoint, test_df(df))






"""# Scaled Linear Tobit"""

objective_lin_NO_trunc = get_objective_fn_tobit_fixed_std(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_LINEAR_TOBIT_SCALED}/{CHECKPOINT_LINEAR_TOBIT_SCALED}', input_size = INPUT_SIZE, is_liniar=True)

def tpe_opt_lin_NO_trunc():
    return tpe_opt_hyperparam(ROOT_LINEAR_TOBIT_SCALED, CHECKPOINT_LINEAR_TOBIT_SCALED, objective_lin_NO_trunc)

def eval_linear_NO_trunc():
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                            ROOT_LINEAR_TOBIT_SCALED, CHECKPOINT_LINEAR_TOBIT_SCALED, is_liniar=True, is_optimized = True)

def plot_linear_NO_trunc():
    checkpoint = load_checkpoint(f'{ROOT_LINEAR_TOBIT_SCALED}/{CHECKPOINT_LINEAR_TOBIT_SCALED} best.tar')
    plot_dataset_and_net(checkpoint, test_df(df), is_liniar=True)






"""# Scaled Linear Tobit With Truncation"""

objective_lin_WITH_trunc = get_objective_fn_tobit_fixed_std(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_LINEAR_TRUNCATED_TOBIT_SCALED}/{CHECKPOINT_LINEAR_TRUNCATED_TOBIT_SCALED}',
    input_size = INPUT_SIZE, is_liniar=True, truncated_low = zero_normalized)


def tpe_opt_lin_WITH_trunc():
    return tpe_opt_hyperparam(ROOT_LINEAR_TRUNCATED_TOBIT_SCALED, CHECKPOINT_LINEAR_TRUNCATED_TOBIT_SCALED, objective_lin_WITH_trunc)

def eval_linear_tobit_WITH_trunc():
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                            ROOT_LINEAR_TRUNCATED_TOBIT_SCALED, CHECKPOINT_LINEAR_TRUNCATED_TOBIT_SCALED, is_liniar=True, is_optimized = True)

def plot_linear_tobit_WITH_trunc():
    checkpoint = load_checkpoint(f'{ROOT_LINEAR_TRUNCATED_TOBIT_SCALED}/{CHECKPOINT_LINEAR_TRUNCATED_TOBIT_SCALED} best.tar')
    plot_dataset_and_net(checkpoint, test_df(df), is_liniar=True)

# eval_deep_tobit_WITH_trunc()
# eval_deep_tobit_NO_trunc()
# eval_linear_tobit_WITH_trunc()
# eval_linear_tobit_NO_trunc()