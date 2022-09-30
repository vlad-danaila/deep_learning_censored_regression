from experiments.synthetic.dynamic.dataset import *
from experiments.synthetic.eval_optimized import plot_and_evaluate_model_mae_mse, plot_dataset_and_net
from experiments.tpe_hyperparam_opt import get_objective_fn_mae_mse, tpe_opt_hyperparam
from experiments.util import TruncatedBetaDistributionConfig, name_from_distribution_config, create_folder
from experiments.synthetic.constants import *
from experiments.synthetic.plot import plot_beta, plot_dataset

def plot_distribution(alpha, beta):
    config = TruncatedBetaDistributionConfig(
        censor_low_bound = CENSOR_LOW_BOUND, censor_high_bound = CENSOR_HIGH_BOUND, alpha = alpha, beta = beta, is_heteroscedastic = False
    )
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(config)
    plot_beta(x_mean, x_std, y_mean, y_std, a=alpha, b=beta)
    plot_dataset(dataset_train)

plot_distribution(1, 4)
