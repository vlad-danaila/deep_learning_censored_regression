from experiments.util import set_random_seed, load_checkpoint
from experiments.real.pm25.dataset import *
from experiments.real.pm25.eval_optimized import plot_and_evaluate_model_tobit_fixed_std, plot_dataset_and_net
from experiments.real.models import get_model, linear_model
from experiments.tpe_hyperparam_opt import get_objective_fn_tobit_fixed_std, tpe_opt_hyperparam

"""Constants"""
ROOT_DEEP_TOBIT_REPARAMETRIZED = 'experiments/real/pm25/tobit_based/reparam_fixed_std/deep_tobit_cens_NO_trunc'
CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED = 'reparametrized deep tobit model'

ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED = 'experiments/real/pm25/tobit_based/reparam_fixed_std/deep_tobit_cens_WITH_trunc'
CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED = 'reparametrized truncated deep tobit model'

ROOT_LINEAR_TOBIT_REPARAMETRIZED = 'experiments/real/pm25/tobit_based/reparam_fixed_std/liniar_tobit_cens_NO_trunc'
CHECKPOINT_LINEAR_TOBIT_REPARAMETRIZED = 'reparametrized linear tobit model'

ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED = 'experiments/real/pm25/tobit_based/reparam_fixed_std/liniar_tobit_cens_WITH_trunc'
CHECKPOINT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED = 'reparametrized truncated linear tobit model'

"""Reproducible experiments"""

set_random_seed()




"""# Reparametrized Deep Tobit"""

objective_deep_NO_trunc = get_objective_fn_tobit_fixed_std(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_DEEP_TOBIT_REPARAMETRIZED}/{CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED}', input_size = INPUT_SIZE, isReparam=True)

def tpe_opt_deep_NO_trunc_reparam():
    return tpe_opt_hyperparam(ROOT_DEEP_TOBIT_REPARAMETRIZED, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED, objective_deep_NO_trunc)

def eval_deep_NO_trunc_reparam():
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                            ROOT_DEEP_TOBIT_REPARAMETRIZED, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED, is_optimized = True)

def plot_deep_NO_trunc_reparam():
    checkpoint = load_checkpoint(f'{ROOT_DEEP_TOBIT_REPARAMETRIZED}/{CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED} best.tar')
    plot_dataset_and_net(checkpoint, test_df(df))




"""# Reparametrized Deep Tobit With Truncation"""

objective_deep_WITH_trunc = get_objective_fn_tobit_fixed_std(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED}/{CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED}', input_size = INPUT_SIZE, truncated_low = zero_normalized, isReparam=True)

def tpe_opt_deep_WITH_trunc_reparam():
    return tpe_opt_hyperparam(ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, objective_deep_WITH_trunc)

def eval_deep_WITH_trunc_reparam():
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                            ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, is_optimized = True)

def plot_deep_tobit_WITH_trunc_reparam():
    checkpoint = load_checkpoint(f'{ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED}/{CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED} best.tar')
    plot_dataset_and_net(checkpoint, test_df(df))





"""# Reparametrized Linear Tobit"""

objective_lin_NO_trunc = get_objective_fn_tobit_fixed_std(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_LINEAR_TOBIT_REPARAMETRIZED}/{CHECKPOINT_LINEAR_TOBIT_REPARAMETRIZED}', input_size = INPUT_SIZE, is_liniar=True, isReparam=True)

def tpe_opt_lin_NO_trunc_reparam():
    return tpe_opt_hyperparam(ROOT_LINEAR_TOBIT_REPARAMETRIZED, CHECKPOINT_LINEAR_TOBIT_REPARAMETRIZED, objective_lin_NO_trunc)

def eval_linear_NO_trunc_reparam():
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                            ROOT_LINEAR_TOBIT_REPARAMETRIZED, CHECKPOINT_LINEAR_TOBIT_REPARAMETRIZED, is_liniar=True, is_optimized = True)

def plot_linear_NO_trunc_reparam():
    checkpoint = load_checkpoint(f'{ROOT_LINEAR_TOBIT_REPARAMETRIZED}/{CHECKPOINT_LINEAR_TOBIT_REPARAMETRIZED} best.tar')
    plot_dataset_and_net(checkpoint, test_df(df), is_liniar=True)






"""# Reparametrized Linear Tobit With Truncation"""

objective_lin_WITH_trunc = get_objective_fn_tobit_fixed_std(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED}/{CHECKPOINT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED}',
    input_size = INPUT_SIZE, is_liniar=True, truncated_low = zero_normalized, isReparam=True)

def tpe_opt_lin_WITH_trunc_reparam():
    return tpe_opt_hyperparam(ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED, CHECKPOINT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED, objective_lin_WITH_trunc)

def eval_linear_tobit_WITH_trunc_reparam():
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                            ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED, CHECKPOINT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED, is_liniar=True, is_optimized = True)

def plot_linear_tobit_WITH_trunc_reparam():
    checkpoint = load_checkpoint(f'{ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED}/{CHECKPOINT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED} best.tar')
    plot_dataset_and_net(checkpoint, test_df(df), is_liniar=True)

# eval_deep_tobit_WITH_trunc_reparam()
# eval_deep_tobit_NO_trunc_reparam()
# eval_linear_tobit_WITH_trunc_reparam()
# eval_linear_tobit_NO_trunc_reparam()