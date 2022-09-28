from experiments.synthetic.dynamic.dataset import *
from experiments.synthetic.eval_optimized import plot_and_evaluate_model_gll
from experiments.synthetic.eval_optimized import plot_dataset_and_net
from experiments.tpe_hyperparam_opt import get_objective_fn_gll, tpe_opt_hyperparam

"""Constants"""
ROOT_GLL = 'experiments/synthetic/heteroscedastic/tobit_based/scaled_fixed_std/gll'
CHECKPOINT_GLL = 'gausian log likelihood model'



"""# PDF Log-Likelihood"""

class GausianLogLikelihoodLoss(t.nn.Module):

    def __init__(self, sigma):
        super(GausianLogLikelihoodLoss, self).__init__()
        self.sigma = sigma
        self.epsilon = t.tensor(1e-40, dtype = t.float32, requires_grad = False, device=get_device())

    def forward(self, y_pred: t.Tensor, y_true: t.Tensor):
        sigma = t.abs(self.sigma)
        return t.sum(t.log(sigma + self.epsilon) + (((y_true - y_pred)/sigma) ** 2) / 2)

def tpe_opt_gll_scaled(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    objective_gll = get_objective_fn_gll(
        dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_GLL}/{CHECKPOINT_GLL}', GausianLogLikelihoodLoss,
        plot = False, log = False)
    return tpe_opt_hyperparam(ROOT_GLL, CHECKPOINT_GLL, objective_gll)

def eval_gll_scaled(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    plot_and_evaluate_model_gll(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                              ROOT_GLL, CHECKPOINT_GLL, GausianLogLikelihoodLoss, is_optimized= True)

def plot_gll_scaled(dataset_config: TruncatedBetaDistributionConfig):
    dataset_train, dataset_val, dataset_test, bound_min, bound_max, zero_normalized, x_mean, x_std, y_mean, y_std = get_experiment_data(dataset_config)
    checkpoint = t.load(f'{ROOT_GLL}/{CHECKPOINT_GLL} best.tar')
    plot_dataset_and_net(checkpoint, x_mean, x_std, y_mean, y_std, dataset_val)