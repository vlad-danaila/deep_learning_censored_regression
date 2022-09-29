from experiments.synthetic.dynamic.dataset import *
from experiments.synthetic.eval_optimized import plot_and_evaluate_model_mae_mse, plot_dataset_and_net
from experiments.tpe_hyperparam_opt import get_objective_fn_mae_mse, tpe_opt_hyperparam
from experiments.util import TruncatedBetaDistributionConfig, name_from_distribution_config, create_folder
from experiments.synthetic.constants import *

"""Constants"""
ROOT_MAE = 'experiments/synthetic/dynamic/mae_based/mae_simple'
CHECKPOINT_MAE = 'mae model'

ROOT_BOUNDED_MAE = 'experiments/synthetic/dynamic/mae_based/mae_cens_NO_trunc'
CHECKPOINT_BOUNDED_MAE = 'mae bounded model'

ROOT_BOUNDED_MAE_WITH_PENALTY = 'experiments/synthetic/dynamic/mae_based/mae_cens_WITH_trunc'
CHECKPOINT_BOUNDED_MAE_WITH_PENALTY = 'mae bounded with penalty model'



"""# MAE"""

"""TPE Hyperparameter Optimisation"""
def tpe_opt_mae_simple(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_MAE + '/' + name_from_distribution_config(dataset_config)
    create_folder(root)
    objective_mae_simple = get_objective_fn_mae_mse(
        dataset_train, dataset_val, bound_min, bound_max, f'{root}/{CHECKPOINT_MAE}', t.nn.L1Loss)
    return tpe_opt_hyperparam(root, CHECKPOINT_MAE, objective_mae_simple)

def eval_mae_simple(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_MAE + '/' + name_from_distribution_config(dataset_config)
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, x_mean, x_std, y_mean, y_std,
                                    dataset_val, dataset_test, root, CHECKPOINT_MAE, t.nn.L1Loss, is_optimized= True)

def plot_mae_simple(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_MAE + '/' + name_from_distribution_config(dataset_config)
    checkpoint = t.load(f'{root}/{CHECKPOINT_MAE} best.tar')
    plot_dataset_and_net(checkpoint, x_mean, x_std, y_mean, y_std, dataset_val)


# default_config = TruncatedBetaDistributionConfig(
#     censor_low_bound = CENSOR_LOW_BOUND, censor_high_bound = CENSOR_HIGH_BOUND, alpha = ALPHA, beta = BETA, is_heteroscedastic = False
# )
# tpe_opt_mae_simple(default_config)
# eval_mae_simple(default_config)
# plot_mae_simple(default_config)



"""# Bounded MAE"""

mae = t.nn.L1Loss()

def get_bounded_loss_lambda(bound_min, bound_max):
    def bounded_loss(y_pred, y):
      y_pred = t.clamp(y_pred, min = bound_min, max = bound_max)
      return mae(y_pred, y)
    return bounded_loss

def tpe_opt_mae_cens_NO_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    bounded_loss = get_bounded_loss_lambda(bound_min, bound_max)
    root = ROOT_BOUNDED_MAE + '/' + name_from_distribution_config(dataset_config)
    create_folder(root)
    objective_mae_bounded = get_objective_fn_mae_mse(
        dataset_train, dataset_val, bound_min, bound_max, root + '/' + CHECKPOINT_BOUNDED_MAE, lambda: bounded_loss)
    return tpe_opt_hyperparam(root, CHECKPOINT_BOUNDED_MAE, objective_mae_bounded)

def eval_mae_cens_NO_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    bounded_loss = get_bounded_loss_lambda(bound_min, bound_max)
    root = ROOT_BOUNDED_MAE + '/' + name_from_distribution_config(dataset_config)
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                    root, CHECKPOINT_BOUNDED_MAE, lambda: bounded_loss, is_optimized = True)

def plot_mae_cens_NO_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_BOUNDED_MAE + '/' + name_from_distribution_config(dataset_config)
    checkpoint = t.load(f'{root}/{CHECKPOINT_BOUNDED_MAE} best.tar')
    plot_dataset_and_net(checkpoint, x_mean, x_std, y_mean, y_std, dataset_val)





"""# Bounded MAE With Penalty"""

def get_bounded_loss_with_penalty_lambda(bound_min, bound_max, zero_normalized):
    bounded_loss = get_bounded_loss_lambda(bound_min, bound_max)
    def below_zero_mae_penalty(y_pred):
        y_below_zero = t.clamp(y_pred, min = -math.inf, max = zero_normalized)
        return mae(y_below_zero, t.full_like(y_below_zero, zero_normalized))
    def bounded_loss_with_penalty(y_pred, y):
        return bounded_loss(y_pred, y) + below_zero_mae_penalty(y_pred)
    return bounded_loss_with_penalty

def tpe_opt_mae_cens_WITH_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    bounded_loss_with_penalty = get_bounded_loss_with_penalty_lambda(bound_min, bound_max, zero_normalized)
    root = ROOT_BOUNDED_MAE_WITH_PENALTY + '/' + name_from_distribution_config(dataset_config)
    create_folder(root)
    objective_mae_bounded_pen = get_objective_fn_mae_mse(dataset_train, dataset_val, bound_min, bound_max,
        root + '/' + CHECKPOINT_BOUNDED_MAE_WITH_PENALTY, lambda: bounded_loss_with_penalty)
    return tpe_opt_hyperparam(root, CHECKPOINT_BOUNDED_MAE_WITH_PENALTY, objective_mae_bounded_pen)

def eval_mae_cens_WITH_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    bounded_loss_with_penalty = get_bounded_loss_with_penalty_lambda(bound_min, bound_max, zero_normalized)
    root = ROOT_BOUNDED_MAE_WITH_PENALTY + '/' + name_from_distribution_config(dataset_config)
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
        root, CHECKPOINT_BOUNDED_MAE_WITH_PENALTY, lambda: bounded_loss_with_penalty, is_optimized= True)

def plot_mae_cens_WITH_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_BOUNDED_MAE_WITH_PENALTY + '/' + name_from_distribution_config(dataset_config)
    checkpoint = t.load(f'{root}/{CHECKPOINT_BOUNDED_MAE_WITH_PENALTY} best.tar')
    plot_dataset_and_net(checkpoint, x_mean, x_std, y_mean, y_std, dataset_val)


# tpe_opt_mae_simple()
# eval_mae_simple()
#
# tpe_opt_mae_cens_NO_trunc()
# eval_mae_cens_NO_trunc()
#
# tpe_opt_mae_cens_WITH_trunc()
# eval_mae_cens_WITH_trunc()