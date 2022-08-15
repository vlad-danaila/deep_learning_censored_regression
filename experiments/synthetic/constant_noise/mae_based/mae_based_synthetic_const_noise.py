from experiments.constants import GRID_RESULTS_FILE
from experiments.synthetic.constants import *
from experiments.util import set_random_seed, get_best_metrics_and_hyperparams_from_optuna_study
from experiments.synthetic.constant_noise.dataset import *
from experiments.grid_search import grid_search, config_validation, get_grid_search_space
from experiments.synthetic.grid_eval import plot_and_evaluate_model_mae_mse
from experiments.grid_train import train_and_evaluate_mae_mse
from experiments.synthetic.grid_eval import plot_dataset_and_net
from experiments.synthetic.models import DenseNetwork
from experiments.tpe_hyperparam_opt import get_objective_fn_mae_mse, tpe_opt_hyperparam

"""Constants"""
ROOT_MAE = 'experiments/synthetic/constant_noise/mae_based/mae_simple'
CHECKPOINT_MAE = 'mae model'

ROOT_BOUNDED_MAE = 'experiments/synthetic/constant_noise/mae_based/mae_cens_NO_trunc'
CHECKPOINT_BOUNDED_MAE = 'mae bounded model'

ROOT_BOUNDED_MAE_WITH_PENALTY = 'experiments/synthetic/constant_noise/mae_based/mae_cens_WITH_trunc'
CHECKPOINT_BOUNDED_MAE_WITH_PENALTY = 'mae bounded with penalty model'

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







"""# MAE"""

objective_mae_mse = get_objective_fn_mae_mse(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_MAE}/{CHECKPOINT_MAE}', t.nn.L1Loss, plot = False, log = False)

"""TPE Hyperparameter Optimisation"""
def tpe_opt_mae_simple():
    best = tpe_opt_hyperparam(ROOT_MAE, CHECKPOINT_MAE, objective_mae_mse)
    return best

def eval_mae_simple():
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, x_mean, x_std, y_mean, y_std,
                                    dataset_val, dataset_test, ROOT_MAE, CHECKPOINT_MAE, t.nn.L1Loss, is_optimized= True)

def plot_mae_simple():
    checkpoint = t.load(f'{ROOT_MAE}/{CHECKPOINT_MAE} best.tar')
    plot_dataset_and_net(checkpoint, DenseNetwork(), x_mean, x_std, y_mean, y_std, dataset_val)

# tpe_opt_mae_simple()
# eval_mae_simple()
# plot_mae_simple()



"""# Bounded MAE"""

mae = t.nn.L1Loss()

def bounded_loss(y_pred, y):
  y_pred = t.clamp(y_pred, min = bound_min, max = bound_max)
  return mae(y_pred, y)

"""### Grid Search"""

train_and_evaluate_net = train_and_evaluate_mae_mse(
    dataset_train, dataset_val, bound_min, bound_max, ROOT_BOUNDED_MAE + '/' + CHECKPOINT_BOUNDED_MAE, lambda: bounded_loss, plot = False, log = False)

def tpe_opt_mae_cens_NO_trunc():
    return tpe_opt_hyperparam(ROOT_BOUNDED_MAE, CHECKPOINT_BOUNDED_MAE, train_and_evaluate_net)


def eval_mae_cens_NO_trunc():
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                    ROOT_BOUNDED_MAE, CHECKPOINT_BOUNDED_MAE, lambda: bounded_loss, is_optimized= True)

def plot_mae_cens_NO_trunc():
    checkpoint = t.load(f'{ROOT_BOUNDED_MAE}/grid {CHECKPOINT_BOUNDED_MAE}.tar')
    plot_dataset_and_net(checkpoint, DenseNetwork(), x_mean, x_std, y_mean, y_std, dataset_val)





"""# Bounded MAE With Penalty"""

def below_zero_mae_penalty(y_pred):
  y_below_zero = t.clamp(y_pred, min = -math.inf, max = zero_normalized)
  return mae(y_below_zero, t.full_like(y_below_zero, zero_normalized))

def bounded_loss_with_penalty(y_pred, y):
  return bounded_loss(y_pred, y) + below_zero_mae_penalty(y_pred)

train_and_evaluate_net = train_and_evaluate_mae_mse(
    dataset_train, dataset_val, bound_min, bound_max,
    ROOT_BOUNDED_MAE_WITH_PENALTY + '/' + CHECKPOINT_BOUNDED_MAE_WITH_PENALTY,
    lambda: bounded_loss_with_penalty, plot = False, log = False)

def grid_search_mae_cens_WITH_trunc():
    return tpe_opt_hyperparam(ROOT_BOUNDED_MAE_WITH_PENALTY, CHECKPOINT_BOUNDED_MAE_WITH_PENALTY, train_and_evaluate_net)

def eval_mae_cens_WITH_trunc():
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test, ROOT_BOUNDED_MAE_WITH_PENALTY,
                                    CHECKPOINT_BOUNDED_MAE_WITH_PENALTY, lambda: bounded_loss_with_penalty, is_optimized= True)

def plot_mae_cens_WITH_trunc():
    checkpoint = t.load(f'{ROOT_BOUNDED_MAE_WITH_PENALTY}/grid {CHECKPOINT_BOUNDED_MAE_WITH_PENALTY}.tar')
    plot_dataset_and_net(checkpoint, DenseNetwork(), x_mean, x_std, y_mean, y_std, dataset_val)