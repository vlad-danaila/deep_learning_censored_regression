from experiments.util import set_random_seed, load_checkpoint
from experiments.real.pm25.dataset import *
from experiments.real.pm25.eval_optimized import plot_and_evaluate_model_mae_mse, plot_dataset_and_net
from experiments.real.models import get_model
from experiments.tpe_hyperparam_opt import get_objective_fn_mae_mse, tpe_opt_hyperparam

"""Constants"""
CHECKPOINT_MSE = 'mse model'
ROOT_MSE = 'experiments/real/pm25/mse_based/mse_simple'

CHECKPOINT_BOUNDED_MSE = 'mse bounded model'
ROOT_BOUNDED_MSE = 'experiments/real/pm25/mse_based/mse_cens_NO_trunc'

CHECKPOINT_BOUNDED_MSE_WITH_PENALTY = 'mse bounded with penalty model'
ROOT_BOUNDED_MSE_WITH_PENALTY = 'experiments/real/pm25/mse_based/mse_cens_WITH_trunc'

"""Reproducible experiments"""

set_random_seed()

"""# MSE"""

objective_mse_simple = get_objective_fn_mae_mse(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_MSE}/{CHECKPOINT_MSE}', t.nn.MSELoss, input_size = INPUT_SIZE)

"""TPE Hyperparameter Optimisation"""
def tpe_opt_mse_simple():
    return tpe_opt_hyperparam(ROOT_MSE, CHECKPOINT_MSE, objective_mse_simple)

def eval_mse_simple():
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, test_df(df),
                                    dataset_val, dataset_test, ROOT_MSE, CHECKPOINT_MSE, t.nn.MSELoss, is_optimized= True)

def plot_mse_simple():
    checkpoint = load_checkpoint(f'{ROOT_MSE}/{CHECKPOINT_MSE} best.tar')
    plot_dataset_and_net(checkpoint, get_model(INPUT_SIZE), test_df(df))

# tpe_opt_mae_simple()
# eval_mae_simple()
# plot_mae_simple()



"""# Bounded MAE"""

mse = t.nn.MSELoss()

def bounded_loss(y_pred, y):
    y_pred = t.clamp(y_pred, min = bound_min, max = bound_max)
    return mse(y_pred, y)

objective_mse_bounded = get_objective_fn_mae_mse(
    dataset_train, dataset_val, bound_min, bound_max, ROOT_BOUNDED_MSE + '/' + CHECKPOINT_BOUNDED_MSE, lambda: bounded_loss, input_size = INPUT_SIZE)

def tpe_opt_mse_cens_NO_trunc():
    return tpe_opt_hyperparam(ROOT_BOUNDED_MSE, CHECKPOINT_BOUNDED_MSE, objective_mse_bounded)

def eval_mse_cens_NO_trunc():
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, test_df(df), dataset_val, dataset_test, ROOT_BOUNDED_MSE,
                                    CHECKPOINT_BOUNDED_MSE, lambda: bounded_loss, is_optimized = True)

def plot_mse_cens_NO_trunc():
    checkpoint = load_checkpoint(f'{ROOT_BOUNDED_MSE}/{CHECKPOINT_BOUNDED_MSE} best.tar')
    plot_dataset_and_net(checkpoint, get_model(INPUT_SIZE), test_df(df))





"""# Bounded MSE With Penalty"""

def below_zero_mse_penalty(y_pred):
    y_below_zero = t.clamp(y_pred, min = -math.inf, max = zero_normalized)
    return mse(y_below_zero, t.full_like(y_below_zero, zero_normalized))

def bounded_loss_with_penalty(y_pred, y):
    return bounded_loss(y_pred, y) + below_zero_mse_penalty(y_pred)

objective_mse_bounded_pen = get_objective_fn_mae_mse(dataset_train, dataset_val, bound_min, bound_max,
                                                     ROOT_BOUNDED_MSE_WITH_PENALTY + '/' + CHECKPOINT_BOUNDED_MSE_WITH_PENALTY, lambda: bounded_loss_with_penalty, input_size = INPUT_SIZE)

def tpe_opt_mse_cens_WITH_trunc():
    return tpe_opt_hyperparam(ROOT_BOUNDED_MSE_WITH_PENALTY, CHECKPOINT_BOUNDED_MSE_WITH_PENALTY, objective_mse_bounded_pen)

def eval_mse_cens_WITH_trunc():
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, test_df(df), dataset_val, dataset_test, ROOT_BOUNDED_MSE_WITH_PENALTY,
                                    CHECKPOINT_BOUNDED_MSE_WITH_PENALTY, lambda: bounded_loss_with_penalty, is_optimized = True)

def plot_mse_cens_WITH_trunc():
    checkpoint = load_checkpoint(f'{ROOT_BOUNDED_MSE_WITH_PENALTY}/{CHECKPOINT_BOUNDED_MSE_WITH_PENALTY} best.tar')
    plot_dataset_and_net(checkpoint, get_model(INPUT_SIZE), test_df(df))

# tpe_opt_mae_simple()
# eval_mae_simple()
#
# tpe_opt_mae_cens_NO_trunc()
# eval_mae_cens_NO_trunc()
#
# tpe_opt_mae_cens_WITH_trunc()
# eval_mae_cens_WITH_trunc()