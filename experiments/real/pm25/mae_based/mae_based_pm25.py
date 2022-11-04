from experiments.util import set_random_seed, load_checkpoint
from experiments.real.pm25.dataset import *
from experiments.real.pm25.eval_optimized import plot_and_evaluate_model_mae_mse, plot_dataset_and_net
from experiments.tpe_hyperparam_opt import get_objective_fn_mae_mse, tpe_opt_hyperparam
from experiments.constants import REAL_EXPERIMENTS_PREFIX
from os import makedirs

"""Constants"""
ROOT_MAE = f'experiments/real/pm25/mae_based/mae_simple/{REAL_EXPERIMENTS_PREFIX}'
CHECKPOINT_MAE = 'mae model'

ROOT_BOUNDED_MAE = f'experiments/real/pm25/mae_based/mae_cens_NO_trunc/{REAL_EXPERIMENTS_PREFIX}'
CHECKPOINT_BOUNDED_MAE = 'mae bounded model'

ROOT_BOUNDED_MAE_WITH_PENALTY = f'experiments/real/pm25/mae_based/mae_cens_WITH_trunc/{REAL_EXPERIMENTS_PREFIX}'
CHECKPOINT_BOUNDED_MAE_WITH_PENALTY = 'mae bounded with penalty model'

for dir in [ROOT_MAE, ROOT_BOUNDED_MAE, ROOT_BOUNDED_MAE_WITH_PENALTY]:
    makedirs(dir, exist_ok=True)

"""Reproducible experiments"""

set_random_seed()

"""# MAE"""

objective_mae_simple = get_objective_fn_mae_mse(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_MAE}/{CHECKPOINT_MAE}', t.nn.L1Loss, input_size = INPUT_SIZE)

"""TPE Hyperparameter Optimisation"""
def tpe_opt_mae_simple():
    return tpe_opt_hyperparam(ROOT_MAE, CHECKPOINT_MAE, objective_mae_simple)

def eval_mae_simple():
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, test_df(df),
                                    dataset_val, dataset_test, ROOT_MAE, CHECKPOINT_MAE, t.nn.L1Loss, is_optimized= True)

def plot_mae_simple():
    checkpoint = load_checkpoint(ROOT_MAE, CHECKPOINT_MAE, is_optimized=True)
    plot_dataset_and_net(checkpoint, test_df(df))

# tpe_opt_mae_simple()
# eval_mae_simple()
# plot_mae_simple()



"""# Bounded MAE"""

mae = t.nn.L1Loss()

def bounded_loss(y_pred, y):
  y_pred = t.clamp(y_pred, min = bound_min, max = bound_max)
  return mae(y_pred, y)

objective_mae_bounded = get_objective_fn_mae_mse(
    dataset_train, dataset_val, bound_min, bound_max, ROOT_BOUNDED_MAE + '/' + CHECKPOINT_BOUNDED_MAE, lambda: bounded_loss, input_size = INPUT_SIZE)

def tpe_opt_mae_cens_NO_trunc():
    return tpe_opt_hyperparam(ROOT_BOUNDED_MAE, CHECKPOINT_BOUNDED_MAE, objective_mae_bounded)

def eval_mae_cens_NO_trunc():
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                    ROOT_BOUNDED_MAE, CHECKPOINT_BOUNDED_MAE, lambda: bounded_loss, is_optimized = True)

def plot_mae_cens_NO_trunc():
    checkpoint = load_checkpoint(ROOT_BOUNDED_MAE, CHECKPOINT_BOUNDED_MAE, is_optimized=True)
    plot_dataset_and_net(checkpoint, test_df(df))





"""# Bounded MAE With Penalty"""

def below_zero_mae_penalty(y_pred):
  y_below_zero = t.clamp(y_pred, min = -math.inf, max = zero_normalized)
  return mae(y_below_zero, t.full_like(y_below_zero, zero_normalized))

def bounded_loss_with_penalty(y_pred, y):
  return bounded_loss(y_pred, y) + below_zero_mae_penalty(y_pred)

objective_mae_bounded_pen = get_objective_fn_mae_mse(dataset_train, dataset_val, bound_min, bound_max,
    ROOT_BOUNDED_MAE_WITH_PENALTY + '/' + CHECKPOINT_BOUNDED_MAE_WITH_PENALTY, lambda: bounded_loss_with_penalty, input_size = INPUT_SIZE)

def tpe_opt_mae_cens_WITH_trunc():
    return tpe_opt_hyperparam(ROOT_BOUNDED_MAE_WITH_PENALTY, CHECKPOINT_BOUNDED_MAE_WITH_PENALTY, objective_mae_bounded_pen)

def eval_mae_cens_WITH_trunc():
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, test_df(df), dataset_val, dataset_test, ROOT_BOUNDED_MAE_WITH_PENALTY,
                                    CHECKPOINT_BOUNDED_MAE_WITH_PENALTY, lambda: bounded_loss_with_penalty, is_optimized= True)

def plot_mae_cens_WITH_trunc():
    checkpoint = load_checkpoint(ROOT_BOUNDED_MAE_WITH_PENALTY, CHECKPOINT_BOUNDED_MAE_WITH_PENALTY, is_optimized=True)
    plot_dataset_and_net(checkpoint, test_df(df))

# tpe_opt_mae_simple()
# eval_mae_simple()
#
# tpe_opt_mae_cens_NO_trunc()
# eval_mae_cens_NO_trunc()
#
# tpe_opt_mae_cens_WITH_trunc()
# eval_mae_cens_WITH_trunc()