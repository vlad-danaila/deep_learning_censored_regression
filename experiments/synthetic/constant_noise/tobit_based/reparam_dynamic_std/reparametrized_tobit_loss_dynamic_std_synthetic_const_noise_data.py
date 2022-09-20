from experiments.synthetic.constants import *
from experiments.util import set_random_seed
from experiments.synthetic.constant_noise.dataset import *
from experiments.synthetic.eval_optimized import plot_and_evaluate_model_tobit_dyn_std, plot_dataset_and_net
from deep_tobit.util import normalize, distinguish_censored_versus_observed_data
from experiments.synthetic.models import DenseNetwork, get_scale_network
from experiments.tpe_hyperparam_opt import get_objective_fn_tobit_dyn_std, tpe_opt_hyperparam

"""Constants"""
ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED = 'experiments/synthetic/constant_noise/tobit_based/reparam_dynamic_std/deep_tobit_cens_WITH_trunc'
CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED = 'reparametrized truncated deep tobit model'

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

"""# Common Tobit Setup"""

censored_collate_fn = distinguish_censored_versus_observed_data(bound_min, bound_max)
uncensored_collate_fn = distinguish_censored_versus_observed_data(-math.inf, math.inf)

tobit_loader_train = t.utils.data.DataLoader(dataset_train, batch_size = 100, shuffle = True, num_workers = 0, collate_fn = censored_collate_fn)
tobit_loader_val = t.utils.data.DataLoader(dataset_val, batch_size = len(dataset_val), shuffle = False, num_workers = 0, collate_fn = censored_collate_fn)
tobit_loader_test = t.utils.data.DataLoader(dataset_test, batch_size = len(dataset_test), shuffle = False, num_workers = 0, collate_fn = uncensored_collate_fn)



"""# Scaled Deep Tobit With Truncation"""

objective_deep_WITH_trunc = get_objective_fn_tobit_dyn_std(dataset_train, dataset_val, bound_min, bound_max,
    f'{ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED}/{CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED}', truncated_low = zero_normalized, is_reparam=True)

def tpe_opt_deep_WITH_trunc_reparam_dyn_std():
    return tpe_opt_hyperparam(ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, objective_deep_WITH_trunc)

def eval_deep_WITH_trunc_reparam_dyn_std():
  plot_and_evaluate_model_tobit_dyn_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                        ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED,
                                        is_optimized= True, is_reparam = True)

def plot_deep_tobit_WITH_trunc_reparam_dyn_std():
    checkpoint = t.load(f'{ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED}/{CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED} best.tar')
    scale_model = get_scale_network()
    scale_model.load_state_dict(checkpoint['gamma'])
    scale_model.eval()
    plot_dataset_and_net(checkpoint, DenseNetwork(), x_mean, x_std, y_mean, y_std, dataset_val, scale_model=scale_model)