from experiments.synthetic.constants import *
from experiments.util import set_random_seed
from experiments.synthetic.heteroscedastic.dataset import *
from experiments.synthetic.eval_optimized import plot_and_evaluate_model_mae_mse
from experiments.synthetic.eval_optimized import plot_dataset_and_net
from experiments.synthetic.models import DenseNetwork
from experiments.tpe_hyperparam_opt import get_objective_fn_mae_mse, tpe_opt_hyperparam

"""Constants"""
ROOT_MSE = 'experiments/synthetic/heteroscedastic/mse_based/mse_simple'
CHECKPOINT_MSE = 'mse model'

ROOT_BOUNDED_MSE = 'experiments/synthetic/heteroscedastic/mse_based/mse_cens_NO_trunc'
CHECKPOINT_BOUNDED_MSE = 'mse bounded model'

ROOT_BOUNDED_MSE_WITH_PENALTY = 'experiments/synthetic/heteroscedastic/mse_based/mse_cens_WITH_trunc'
CHECKPOINT_BOUNDED_MSE_WITH_PENALTY = 'mse bounded with penalty model'


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




"""# MSE"""

objective_mse_simple = get_objective_fn_mae_mse(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_MSE}/{CHECKPOINT_MSE}', t.nn.MSELoss)

"""TPE Hyperparameter Optimisation"""
def tpe_opt_mse_simple():
    return tpe_opt_hyperparam(ROOT_MSE, CHECKPOINT_MSE, objective_mse_simple)

def eval_mse_simple():
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, x_mean, x_std, y_mean, y_std,
                                    dataset_val, dataset_test, ROOT_MSE, CHECKPOINT_MSE, t.nn.MSELoss, is_optimized= True)

def plot_mse_simple():
    checkpoint = t.load(f'{ROOT_MSE}/{CHECKPOINT_MSE} best.tar')
    plot_dataset_and_net(checkpoint, DenseNetwork(), x_mean, x_std, y_mean, y_std, dataset_val)

# tpe_opt_mse_simple()
# eval_mse_simple()
# plot_mse_simple()



"""# Bounded MSE"""

mse = t.nn.MSELoss()

def bounded_loss(y_pred, y):
  y_pred = t.clamp(y_pred, min = bound_min, max = bound_max)
  return mse(y_pred, y)

objective_mse_bounded = get_objective_fn_mae_mse(
    dataset_train, dataset_val, bound_min, bound_max, ROOT_BOUNDED_MSE + '/' + CHECKPOINT_BOUNDED_MSE, lambda: bounded_loss)

def tpe_opt_mse_cens_NO_trunc():
    best = tpe_opt_hyperparam(ROOT_BOUNDED_MSE, CHECKPOINT_BOUNDED_MSE, objective_mse_bounded)
    return best

def eval_mse_cens_NO_trunc():
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                    ROOT_BOUNDED_MSE, CHECKPOINT_BOUNDED_MSE, lambda: bounded_loss, is_optimized= True)

def plot_mse_cens_NO_trunc():
    checkpoint = t.load(f'{ROOT_BOUNDED_MSE}/{CHECKPOINT_BOUNDED_MSE} best.tar')
    plot_dataset_and_net(checkpoint, DenseNetwork(), x_mean, x_std, y_mean, y_std, dataset_val)





"""# Bounded MSE With Penalty"""

def below_zero_mse_penalty(y_pred):
  y_below_zero = t.clamp(y_pred, min = -math.inf, max = zero_normalized)
  return mse(y_below_zero, t.full_like(y_below_zero, zero_normalized))

def bounded_loss_with_penalty(y_pred, y):
  return bounded_loss(y_pred, y) + below_zero_mse_penalty(y_pred)

objective_mse_bounded_pen = get_objective_fn_mae_mse(
    dataset_train, dataset_val, bound_min, bound_max,
    ROOT_BOUNDED_MSE_WITH_PENALTY + '/' + CHECKPOINT_BOUNDED_MSE_WITH_PENALTY,
    lambda: bounded_loss_with_penalty, plot = False, log = False)

def tpe_opt_mse_cens_WITH_trunc():
    return tpe_opt_hyperparam(ROOT_BOUNDED_MSE_WITH_PENALTY, CHECKPOINT_BOUNDED_MSE_WITH_PENALTY, objective_mse_bounded_pen)

def eval_mse_cens_WITH_trunc():
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test, ROOT_BOUNDED_MSE_WITH_PENALTY,
                                    CHECKPOINT_BOUNDED_MSE_WITH_PENALTY, lambda: bounded_loss_with_penalty, is_optimized= True)

def plot_mse_cens_WITH_trunc():
    checkpoint = t.load(f'{ROOT_BOUNDED_MSE_WITH_PENALTY}/{CHECKPOINT_BOUNDED_MSE_WITH_PENALTY} best.tar')
    plot_dataset_and_net(checkpoint, DenseNetwork(), x_mean, x_std, y_mean, y_std, dataset_val)


# tpe_opt_mse_simple()
# eval_mse_simple()
#
# tpe_opt_mse_cens_NO_trunc()
# eval_mse_cens_NO_trunc()
#
# tpe_opt_mse_cens_WITH_trunc()
# eval_mse_cens_WITH_trunc()