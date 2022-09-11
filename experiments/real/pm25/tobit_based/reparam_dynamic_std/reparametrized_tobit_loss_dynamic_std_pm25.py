from experiments.util import set_random_seed, load_checkpoint
from experiments.real.pm25.dataset import *
from experiments.real.pm25.eval_optimized import plot_and_evaluate_model_tobit_dyn_std, plot_dataset_and_net
from experiments.real.models import get_model, linear_model, get_scale_network
from experiments.tpe_hyperparam_opt import get_objective_fn_tobit_dyn_std, tpe_opt_hyperparam

"""Constants"""
ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED = 'experiments/real/pm25/tobit_based/reparam_dynamic_std/deep_tobit_cens_WITH_trunc'
CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED = 'reparametrized truncated deep tobit model'

"""Reproducible experiments"""

set_random_seed()




"""# Scaled Deep Tobit With Truncation"""
objective_deep_WITH_trunc = get_objective_fn_tobit_dyn_std(dataset_train, dataset_val, bound_min, bound_max,
    f'{ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED}/{CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED}',
    model_fn = lambda: get_model(INPUT_SIZE),
    scale_model_fn = lambda: get_scale_network(INPUT_SIZE),
    truncated_low = zero_normalized,
    is_reparam=True)

def tpe_opt_deep_WITH_trunc_dyn_std_reparam():
    return tpe_opt_hyperparam(ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, objective_deep_WITH_trunc)

def eval_deep_reparam_tobit_WITH_trunc_dyn_std():
    plot_and_evaluate_model_tobit_dyn_std(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                          ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, is_optimized = True, is_reparam = True)

def plot_deep_reparam_tobit_WITH_trunc_dyn_std():
    checkpoint = load_checkpoint(f'{ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED}/{CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED} best.tar')
    scale_model = get_scale_network(INPUT_SIZE)
    scale_model.load_state_dict(checkpoint['gamma'])
    scale_model.eval()
    plot_dataset_and_net(checkpoint, get_model(INPUT_SIZE), test_df(df), scale_model=scale_model)