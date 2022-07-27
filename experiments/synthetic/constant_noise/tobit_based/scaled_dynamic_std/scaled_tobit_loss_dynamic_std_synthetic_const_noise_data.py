from experiments.synthetic.constants import *
from experiments.util import set_random_seed
from experiments.synthetic.constant_noise.dataset import *
from experiments.synthetic.grid_search import train_and_evaluate_tobit_dyn_std, plot_and_evaluate_model_tobit_dyn_std, grid_search, config_validation, get_grid_search_space
from deep_tobit.util import normalize, distinguish_censored_versus_observed_data
from experiments.models import DenseNetwork

"""Constants"""
ROOT_DEEP_TOBIT_TRUNCATED = 'experiments/synthetic/constant_noise/tobit_based/scaled_dynamic_std/deep_tobit_cens_WITH_trunc'
CHECKPOINT_DEEP_TOBIT_TRUNCATED = 'scaled truncated deep tobit model'

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

train_and_evaluate_net = train_and_evaluate_tobit_dyn_std(ROOT_DEEP_TOBIT_TRUNCATED + '/' + CHECKPOINT_DEEP_TOBIT_TRUNCATED,
                                                            model_fn = DenseNetwork, plot = False, log = False, truncated_low = zero_normalized)

def train_once_deep_tobit_WITH_trunc():
    conf = {
        'anneal_strategy': 'linear',
        'base_momentum': 0.85,
        'batch': 100,
        'div_factor': 5,
        'epochs': 20,
        'final_div_factor': 10000.0,
        'grad_clip': 1e-2,
        'max_lr': 5e-3,
        'max_momentum': 0.95,
        'pct_start': 0.45,
        'weight_decay': 0
    }
    train_and_evaluate_net(dataset_train, dataset_val, bound_min, bound_max, conf)
    plot_and_evaluate_model_tobit_dyn_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                          ROOT_DEEP_TOBIT_TRUNCATED, CHECKPOINT_DEEP_TOBIT_TRUNCATED, model_fn = DenseNetwork, isGrid = False)

def grid_search_deep_tobit_WITH_trunc():
    grid_config = [{
        'max_lr': [1e-5, ],
        'epochs': [10],
        'batch': [ 200],
        'pct_start': [0.45],
        'anneal_strategy': ['linear'],
        'base_momentum': [0.85],
        'max_momentum': [0.95],
        'div_factor': [10],
        'final_div_factor': [1e4],
        'weight_decay': [0],
        'grad_clip': [10, ]
    }]
    # grid_config = [{
    #     'max_lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
    #     'epochs': [10, 20],
    #     'batch': [100, 200],
    #     'pct_start': [0.45],
    #     'anneal_strategy': ['linear'],
    #     'base_momentum': [0.85],
    #     'max_momentum': [0.95],
    #     'div_factor': [10, 5, 2],
    #     'final_div_factor': [1e4],
    #     'weight_decay': [0],
    #     'grad_clip': [1e-2, 1e-1, 1, 10, 100]
    # }]
    grid_best = grid_search(ROOT_DEEP_TOBIT_TRUNCATED, dataset_train, dataset_val, bound_min, bound_max,
                            grid_config, train_and_evaluate_net, CHECKPOINT_DEEP_TOBIT_TRUNCATED, conf_validation = config_validation)
    return grid_best

def eval_deep_tobit_WITH_trunc():
    plot_and_evaluate_model_tobit_dyn_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                          ROOT_DEEP_TOBIT_TRUNCATED, CHECKPOINT_DEEP_TOBIT_TRUNCATED, model_fn = DenseNetwork, isGrid = True)
    grid_results = t.load(ROOT_DEEP_TOBIT_TRUNCATED + '/' + GRID_RESULTS_FILE)
    best_config = grid_results['best']
    best_metrics = grid_results[str(best_config)]
    print(best_config)
    print(best_metrics)

# train_once_deep_tobit_WITH_trunc()
# grid_search_deep_tobit_WITH_trunc()
eval_deep_tobit_WITH_trunc()
