from experiments.util import get_scale_model_from_checkpoint, name_from_distribution_config, create_folder
from experiments.synthetic.dynamic.dataset import *
from experiments.synthetic.eval_optimized import plot_and_evaluate_model_tobit_dyn_std, plot_dataset_and_net
from experiments.tpe_hyperparam_opt import get_objective_fn_tobit_dyn_std, tpe_opt_hyperparam

"""Constants"""
ROOT_DEEP_TOBIT_TRUNCATED = 'experiments/synthetic/heteroscedastic/tobit_based/scaled_dynamic_std/deep_tobit_cens_WITH_trunc'
CHECKPOINT_DEEP_TOBIT_TRUNCATED = 'scaled truncated deep tobit model'




"""# Scaled Deep Tobit With Truncation"""

def tpe_opt_deep_WITH_trunc_dyn_std(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_DEEP_TOBIT_TRUNCATED + '/' + name_from_distribution_config(dataset_config)
    create_folder(root)
    objective_deep_WITH_trunc = get_objective_fn_tobit_dyn_std(dataset_train, dataset_val, bound_min, bound_max,
        f'{root}/{CHECKPOINT_DEEP_TOBIT_TRUNCATED}', truncated_low = zero_normalized)
    return tpe_opt_hyperparam(root, CHECKPOINT_DEEP_TOBIT_TRUNCATED, objective_deep_WITH_trunc,
        n_trials = 5000, n_startup_trials = 2500, prunner_warmup_trials = 2500)

def eval_deep_WITH_trunc_dyn_std(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_DEEP_TOBIT_TRUNCATED + '/' + name_from_distribution_config(dataset_config)
    plot_and_evaluate_model_tobit_dyn_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                          root, CHECKPOINT_DEEP_TOBIT_TRUNCATED, is_optimized= True)

def plot_deep_tobit_WITH_trunc_dyn_std(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_DEEP_TOBIT_TRUNCATED + '/' + name_from_distribution_config(dataset_config)
    checkpoint = t.load(f'{root}/{CHECKPOINT_DEEP_TOBIT_TRUNCATED} best.tar')
    scale_model = get_scale_model_from_checkpoint(1, checkpoint)
    scale_model.load_state_dict(checkpoint['sigma'])
    scale_model.eval()
    plot_dataset_and_net(checkpoint, x_mean, x_std, y_mean, y_std, dataset_val, scale_model=scale_model)


