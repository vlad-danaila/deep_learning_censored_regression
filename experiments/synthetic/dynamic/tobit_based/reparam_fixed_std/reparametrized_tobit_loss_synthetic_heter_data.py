from experiments.synthetic.dynamic.dataset import *
from experiments.synthetic.eval_optimized import plot_and_evaluate_model_tobit_fixed_std
from experiments.synthetic.eval_optimized import plot_dataset_and_net
from experiments.tpe_hyperparam_opt import get_objective_fn_tobit_fixed_std, tpe_opt_hyperparam
from experiments.util import TruncatedBetaDistributionConfig

"""Constants"""
ROOT_DEEP_TOBIT_REPARAMETRIZED = 'experiments/synthetic/heteroscedastic/tobit_based/reparam_fixed_std/deep_tobit_cens_NO_trunc'
CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED = 'reparametrized deep tobit model'

ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED = 'experiments/synthetic/heteroscedastic/tobit_based/reparam_fixed_std/deep_tobit_cens_WITH_trunc'
CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED = 'reparametrized truncated deep tobit model'

ROOT_LINEAR_TOBIT_REPARAMETRIZED = 'experiments/synthetic/heteroscedastic/tobit_based/reparam_fixed_std/liniar_tobit_NO_trunc'
CHECKPOINT_LINEAR_TOBIT_REPARAMETRIZED = 'reparametrized linear tobit model'

ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED = 'experiments/synthetic/heteroscedastic/tobit_based/reparam_fixed_std/liniar_tobit_WITH_trunc'
CHECKPOINT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED = 'reparametrized truncated linear tobit model'





"""# Reparametrized Deep Tobit"""


def tpe_opt_deep_NO_trunc_reparam(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    objective_deep_NO_trunc = get_objective_fn_tobit_fixed_std(
        dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_DEEP_TOBIT_REPARAMETRIZED}/{CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED}',
        plot = False, log = False, isReparam=True)
    return tpe_opt_hyperparam(ROOT_DEEP_TOBIT_REPARAMETRIZED, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED, objective_deep_NO_trunc)

def eval_deep_NO_trunc_reparam(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                          ROOT_DEEP_TOBIT_REPARAMETRIZED, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED, is_optimized= True)

def plot_deep_tobit_NO_trunc_reparam(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    checkpoint = t.load(f'{ROOT_DEEP_TOBIT_REPARAMETRIZED}/{CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED} best.tar')
    plot_dataset_and_net(checkpoint, x_mean, x_std, y_mean, y_std, dataset_val)







"""# Reparametrized Deep Tobit With Truncation"""

def tpe_opt_deep_WITH_trunc_reparam(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    objective_deep_WITH_trunc = get_objective_fn_tobit_fixed_std(
        dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED}/{CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED}',
        plot = False, log = False, truncated_low = zero_normalized, isReparam=True)
    return tpe_opt_hyperparam(ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, objective_deep_WITH_trunc)

def eval_deep_WITH_trunc_reparam(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                          ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, is_optimized= True)

def plot_deep_WITH_trunc_reparam(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    checkpoint = t.load(f'{ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED}/{CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED} best.tar')
    plot_dataset_and_net(checkpoint, x_mean, x_std, y_mean, y_std, dataset_val)






"""# Reparametrized Linear Tobit"""

def tpe_opt_lin_NO_trunc_reparam(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    objective_lin_NO_trunc = get_objective_fn_tobit_fixed_std(
        dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_LINEAR_TOBIT_REPARAMETRIZED}/{CHECKPOINT_LINEAR_TOBIT_REPARAMETRIZED}',
        is_liniar = True, plot = False, log = False, isReparam=True)
    return tpe_opt_hyperparam(ROOT_LINEAR_TOBIT_REPARAMETRIZED, CHECKPOINT_LINEAR_TOBIT_REPARAMETRIZED, objective_lin_NO_trunc)

def eval_linear_tobit_NO_trunc_reparam(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                          ROOT_LINEAR_TOBIT_REPARAMETRIZED, CHECKPOINT_LINEAR_TOBIT_REPARAMETRIZED, is_liniar=True, is_optimized= True)

def plot_linear_tobit_NO_trunc_reparam(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    checkpoint = t.load(f'{ROOT_LINEAR_TOBIT_REPARAMETRIZED}/{CHECKPOINT_LINEAR_TOBIT_REPARAMETRIZED} best.tar')
    plot_dataset_and_net(checkpoint, x_mean, x_std, y_mean, y_std, dataset_val, is_liniar=True)







"""# Reparametrized Linear Tobit With Truncation"""

def tpe_opt_lin_WITH_trunc_reparam(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    objective_lin_WITH_trunc = get_objective_fn_tobit_fixed_std(
        dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED}/{CHECKPOINT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED}',
        is_liniar = True, plot = False, log = False, truncated_low = zero_normalized, isReparam=True)
    return tpe_opt_hyperparam(ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED, CHECKPOINT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED, objective_lin_WITH_trunc)

def eval_linear_tobit_WITH_trunc_reparam(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                          ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED, CHECKPOINT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED, is_liniar=True, is_optimized= True)

def plot_linear_tobit_WITH_trunc_reparam(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    checkpoint = t.load(f'{ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED}/{CHECKPOINT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED} best.tar')
    plot_dataset_and_net(checkpoint, x_mean, x_std, y_mean, y_std, dataset_val, is_liniar=True)
