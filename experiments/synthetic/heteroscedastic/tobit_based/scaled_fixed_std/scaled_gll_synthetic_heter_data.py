from experiments.synthetic.constants import *
from experiments.util import set_random_seed
from experiments.synthetic.heteroscedastic.dataset import *
from experiments.synthetic.grid_search import train_and_evaluate_gll, plot_and_evaluate_model_gll, grid_search, config_validation, get_grid_search_space

"""Constants"""
ROOT_GLL = 'experiments/synthetic/heteroscedastic/tobit_based/scaled_fixed_std/gll'
CHECKPOINT_GLL = 'gausian log likelihood model'

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


"""# PDF Log-Likelihood"""

class GausianLogLikelihoodLoss(t.nn.Module):

    def __init__(self, sigma):
        super(GausianLogLikelihoodLoss, self).__init__()
        self.sigma = sigma
        self.epsilon = t.tensor(1e-40, dtype = t.float32, requires_grad = False)

    def forward(self, y_pred: t.Tensor, y_true: t.Tensor):
        sigma = t.abs(self.sigma)
        return t.sum(t.log(sigma + self.epsilon) + (((y_true - y_pred)/sigma) ** 2) / 2)

"""### Grid Search"""

train_and_evaluate_net = train_and_evaluate_gll(ROOT_GLL + '/' + CHECKPOINT_GLL, GausianLogLikelihoodLoss, plot = False, log = False)

def train_once_gll():
    conf = {
        'max_lr': 1e-4,
        'epochs': 10,
        'batch': 100,
        'pct_start': 0.3,
        'anneal_strategy': 'linear',
        'base_momentum': 0.85,
        'max_momentum': 0.95,
        'div_factor': 5,
        'final_div_factor': 1e4,
        'weight_decay': 0
    }
    train_and_evaluate_net(dataset_train, dataset_val, bound_min, bound_max, conf)
    plot_and_evaluate_model_gll(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                    ROOT_GLL, CHECKPOINT_GLL, GausianLogLikelihoodLoss, isGrid = False)

def grid_search_gll():
    grid_config = get_grid_search_space()
    grid_best = grid_search(ROOT_GLL, dataset_train, dataset_val, bound_min, bound_max,
                            grid_config, train_and_evaluate_net, CHECKPOINT_GLL, conf_validation = config_validation)
    return grid_best

def eval_gll_scaled():
    plot_and_evaluate_model_gll(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                    ROOT_GLL, CHECKPOINT_GLL, GausianLogLikelihoodLoss, isGrid = True)
    grid_results = t.load(ROOT_GLL + '/' + GRID_RESULTS_FILE)
    best_config = grid_results['best']
    best_metrics = grid_results[str(best_config)]
    print(best_config)
    print(best_metrics)

eval_gll_scaled()