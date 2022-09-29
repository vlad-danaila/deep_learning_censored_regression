from experiments.synthetic.dynamic.dataset import *
from experiments.synthetic.eval_optimized import plot_and_evaluate_model_mae_mse
from experiments.synthetic.eval_optimized import plot_dataset_and_net
from experiments.tpe_hyperparam_opt import get_objective_fn_mae_mse, tpe_opt_hyperparam
from experiments.util import TruncatedBetaDistributionConfig, name_from_distribution_config, create_folder

"""Constants"""
ROOT_MSE = 'experiments/synthetic/heteroscedastic/mse_based/mse_simple'
CHECKPOINT_MSE = 'mse model'

ROOT_BOUNDED_MSE = 'experiments/synthetic/heteroscedastic/mse_based/mse_cens_NO_trunc'
CHECKPOINT_BOUNDED_MSE = 'mse bounded model'

ROOT_BOUNDED_MSE_WITH_PENALTY = 'experiments/synthetic/heteroscedastic/mse_based/mse_cens_WITH_trunc'
CHECKPOINT_BOUNDED_MSE_WITH_PENALTY = 'mse bounded with penalty model'




"""# MSE"""

"""TPE Hyperparameter Optimisation"""
def tpe_opt_mse_simple(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_MSE + '/' + name_from_distribution_config(dataset_config)
    create_folder(root)
    objective_mse_simple = get_objective_fn_mae_mse(
        dataset_train, dataset_val, bound_min, bound_max, f'{root}/{CHECKPOINT_MSE}', t.nn.MSELoss)
    return tpe_opt_hyperparam(root, CHECKPOINT_MSE, objective_mse_simple)

def eval_mse_simple(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_MSE + '/' + name_from_distribution_config(dataset_config)
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, x_mean, x_std, y_mean, y_std,
                                    dataset_val, dataset_test, root, CHECKPOINT_MSE, t.nn.MSELoss, is_optimized= True)

def plot_mse_simple(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_MSE + '/' + name_from_distribution_config(dataset_config)
    checkpoint = t.load(f'{root}/{CHECKPOINT_MSE} best.tar')
    plot_dataset_and_net(checkpoint, x_mean, x_std, y_mean, y_std, dataset_val)

# tpe_opt_mse_simple()
# eval_mse_simple()
# plot_mse_simple()



"""# Bounded MSE"""

mse = t.nn.MSELoss()

def get_bounded_loss_lambda(bound_min, bound_max):
    def bounded_loss(y_pred, y):
      y_pred = t.clamp(y_pred, min = bound_min, max = bound_max)
      return mse(y_pred, y)
    return bounded_loss

def tpe_opt_mse_cens_NO_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    bounded_loss = get_bounded_loss_lambda(bound_min, bound_max)
    root = ROOT_BOUNDED_MSE + '/' + name_from_distribution_config(dataset_config)
    create_folder(root)
    objective_mse_bounded = get_objective_fn_mae_mse(
        dataset_train, dataset_val, bound_min, bound_max, root + '/' + CHECKPOINT_BOUNDED_MSE, lambda: bounded_loss)
    best = tpe_opt_hyperparam(root, CHECKPOINT_BOUNDED_MSE, objective_mse_bounded)
    return best

def eval_mse_cens_NO_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    bounded_loss = get_bounded_loss_lambda(bound_min, bound_max)
    root = ROOT_BOUNDED_MSE + '/' + name_from_distribution_config(dataset_config)
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                    root, CHECKPOINT_BOUNDED_MSE, lambda: bounded_loss, is_optimized= True)

def plot_mse_cens_NO_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_BOUNDED_MSE + '/' + name_from_distribution_config(dataset_config)
    checkpoint = t.load(f'{root}/{CHECKPOINT_BOUNDED_MSE} best.tar')
    plot_dataset_and_net(checkpoint, x_mean, x_std, y_mean, y_std, dataset_val)





"""# Bounded MSE With Penalty"""

def get_bounded_loss_with_penalty_lambda(bound_min, bound_max, zero_normalized):
    bounded_loss = get_bounded_loss_lambda(bound_min, bound_max)
    def below_zero_mse_penalty(y_pred):
        y_below_zero = t.clamp(y_pred, min = -math.inf, max = zero_normalized)
        return mse(y_below_zero, t.full_like(y_below_zero, zero_normalized))
    def bounded_loss_with_penalty(y_pred, y):
        return bounded_loss(y_pred, y) + below_zero_mse_penalty(y_pred)
    return bounded_loss_with_penalty

def tpe_opt_mse_cens_WITH_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    bounded_loss_with_penalty = get_bounded_loss_with_penalty_lambda(bound_min, bound_max, zero_normalized)
    root = ROOT_BOUNDED_MSE_WITH_PENALTY + '/' + name_from_distribution_config(dataset_config)
    create_folder(root)
    objective_mse_bounded_pen = get_objective_fn_mae_mse(
        dataset_train, dataset_val, bound_min, bound_max,
        root + '/' + CHECKPOINT_BOUNDED_MSE_WITH_PENALTY,
        lambda: bounded_loss_with_penalty, plot = False, log = False)
    return tpe_opt_hyperparam(root, CHECKPOINT_BOUNDED_MSE_WITH_PENALTY, objective_mse_bounded_pen)

def eval_mse_cens_WITH_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    bounded_loss_with_penalty = get_bounded_loss_with_penalty_lambda(bound_min, bound_max, zero_normalized)
    root = ROOT_BOUNDED_MSE_WITH_PENALTY + '/' + name_from_distribution_config(dataset_config)
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                root, CHECKPOINT_BOUNDED_MSE_WITH_PENALTY, lambda: bounded_loss_with_penalty, is_optimized= True)

def plot_mse_cens_WITH_trunc(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    root = ROOT_BOUNDED_MSE_WITH_PENALTY + '/' + name_from_distribution_config(dataset_config)
    checkpoint = t.load(f'{root}/{CHECKPOINT_BOUNDED_MSE_WITH_PENALTY} best.tar')
    plot_dataset_and_net(checkpoint, x_mean, x_std, y_mean, y_std, dataset_val)


# tpe_opt_mse_simple()
# eval_mse_simple()
#
# tpe_opt_mse_cens_NO_trunc()
# eval_mse_cens_NO_trunc()
#
# tpe_opt_mse_cens_WITH_trunc()
# eval_mse_cens_WITH_trunc()