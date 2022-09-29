from experiments.synthetic.dynamic.dataset import *
from experiments.synthetic.eval_optimized import plot_and_evaluate_model_tobit_fixed_std
from experiments.synthetic.eval_optimized import plot_dataset_and_net
from experiments.tpe_hyperparam_opt import get_objective_fn_tobit_fixed_std, tpe_opt_hyperparam
from experiments.util import TruncatedBetaDistributionConfig, name_from_distribution_config, create_folder

"""Constants"""
ROOT_DEEP_TOBIT_SCALED = 'experiments/synthetic/heteroscedastic/tobit_based/scaled_fixed_std/deep_tobit_cens_NO_trunc'
CHECKPOINT_DEEP_TOBIT_SCALED = 'scaled deep tobit model'

ROOT_DEEP_TOBIT_SCALED_TRUNCATED = 'experiments/synthetic/heteroscedastic/tobit_based/scaled_fixed_std/deep_tobit_cens_WITH_trunc'
CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED = 'scaled truncated deep tobit model'

ROOT_LINEAR_TOBIT_SCALED = 'experiments/synthetic/heteroscedastic/tobit_based/scaled_fixed_std/liniar_tobit_cens_NO_trunc'
CHECKPOINT_LINEAR_TOBIT_SCALED = 'scaled linear tobit model'

ROOT_LINEAR_TRUNCATED_TOBIT_SCALED = 'experiments/synthetic/heteroscedastic/tobit_based/scaled_fixed_std/liniar_tobit_cens_WITH_trunc'
CHECKPOINT_LINEAR_TRUNCATED_TOBIT_SCALED = 'scaled truncated linear tobit model'





"""# Scaled Deep Tobit"""

def tpe_opt_deep_NO_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_DEEP_TOBIT_SCALED + '/' + name_from_distribution_config(dataset_config)
    create_folder(root)
    objective_deep_NO_trunc = get_objective_fn_tobit_fixed_std(
        dataset_train, dataset_val, bound_min, bound_max, f'{root}/{CHECKPOINT_DEEP_TOBIT_SCALED}',
        plot = False, log = False)
    return tpe_opt_hyperparam(root, CHECKPOINT_DEEP_TOBIT_SCALED, objective_deep_NO_trunc)

def eval_deep_NO_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_DEEP_TOBIT_SCALED + '/' + name_from_distribution_config(dataset_config)
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                            root, CHECKPOINT_DEEP_TOBIT_SCALED, is_optimized= True)

def plot_deep_NO_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_DEEP_TOBIT_SCALED + '/' + name_from_distribution_config(dataset_config)
    checkpoint = t.load(f'{root}/{CHECKPOINT_DEEP_TOBIT_SCALED} best.tar')
    plot_dataset_and_net(checkpoint, x_mean, x_std, y_mean, y_std, dataset_val)





"""# Scaled Deep Tobit With Truncation"""

def tpe_opt_deep_WITH_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_DEEP_TOBIT_SCALED_TRUNCATED + '/' + name_from_distribution_config(dataset_config)
    create_folder(root)
    objective_deep_WITH_trunc = get_objective_fn_tobit_fixed_std(
        dataset_train, dataset_val, bound_min, bound_max, f'{root}/{CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED}',
        plot = False, log = False, truncated_low = zero_normalized)
    return tpe_opt_hyperparam(root, CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED, objective_deep_WITH_trunc)

def eval_deep_WITH_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_DEEP_TOBIT_SCALED_TRUNCATED + '/' + name_from_distribution_config(dataset_config)
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                            root, CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED, is_optimized= True)

def plot_deep_WITH_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_DEEP_TOBIT_SCALED_TRUNCATED + '/' + name_from_distribution_config(dataset_config)
    checkpoint = t.load(f'{root}/{CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED} best.tar')
    plot_dataset_and_net(checkpoint, x_mean, x_std, y_mean, y_std, dataset_val)






"""# Scaled Linear Tobit"""

def tpe_opt_lin_NO_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_LINEAR_TOBIT_SCALED + '/' + name_from_distribution_config(dataset_config)
    create_folder(root)
    objective_lin_NO_trunc = get_objective_fn_tobit_fixed_std(
        dataset_train, dataset_val, bound_min, bound_max, f'{root}/{CHECKPOINT_LINEAR_TOBIT_SCALED}',
        is_liniar = True, plot = False, log = False)
    return tpe_opt_hyperparam(root, CHECKPOINT_LINEAR_TOBIT_SCALED, objective_lin_NO_trunc)

def eval_lin_NO_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_LINEAR_TOBIT_SCALED + '/' + name_from_distribution_config(dataset_config)
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                            root, CHECKPOINT_LINEAR_TOBIT_SCALED, is_liniar=True, is_optimized= True)

def plot_lin_NO_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_LINEAR_TOBIT_SCALED + '/' + name_from_distribution_config(dataset_config)
    checkpoint = t.load(f'{root}/{CHECKPOINT_LINEAR_TOBIT_SCALED} best.tar')
    plot_dataset_and_net(checkpoint, x_mean, x_std, y_mean, y_std, dataset_val, is_liniar=True)






"""# Scaled Linear Tobit With Truncation"""

def tpe_opt_lin_WITH_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_LINEAR_TRUNCATED_TOBIT_SCALED + '/' + name_from_distribution_config(dataset_config)
    create_folder(root)
    objective_lin_WITH_trunc = get_objective_fn_tobit_fixed_std(
        dataset_train, dataset_val, bound_min, bound_max, f'{root}/{CHECKPOINT_LINEAR_TRUNCATED_TOBIT_SCALED}',
        is_liniar = True, plot = False, log = False, truncated_low = zero_normalized)
    return tpe_opt_hyperparam(root, CHECKPOINT_LINEAR_TRUNCATED_TOBIT_SCALED, objective_lin_WITH_trunc)

def eval_linear_tobit_WITH_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_LINEAR_TRUNCATED_TOBIT_SCALED + '/' + name_from_distribution_config(dataset_config)
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                            root, CHECKPOINT_LINEAR_TRUNCATED_TOBIT_SCALED, is_liniar=True, is_optimized= True)

def plot_linear_tobit_WITH_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_LINEAR_TRUNCATED_TOBIT_SCALED + '/' + name_from_distribution_config(dataset_config)
    checkpoint = t.load(f'{root}/{CHECKPOINT_LINEAR_TRUNCATED_TOBIT_SCALED} best.tar')
    plot_dataset_and_net(checkpoint, x_mean, x_std, y_mean, y_std, dataset_val, is_liniar=True)