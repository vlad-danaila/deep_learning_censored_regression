from experiments.synthetic.dynamic.mae_based.mae_based_synthetic_heter import tpe_opt_mae_simple, tpe_opt_mae_cens_NO_trunc, tpe_opt_mae_cens_WITH_trunc
from experiments.synthetic.dynamic.mse_based.mse_based_synthetic_heter import tpe_opt_mse_simple, tpe_opt_mse_cens_NO_trunc, tpe_opt_mse_cens_WITH_trunc
from experiments.synthetic.dynamic.tobit_based.scaled_fixed_std.scaled_gll_synthetic_heter_data import tpe_opt_gll_scaled
from experiments.synthetic.dynamic.tobit_based.scaled_fixed_std.scaled_tobit_loss_synthetic_heter_data import tpe_opt_deep_NO_trunc, tpe_opt_deep_WITH_trunc, \
    tpe_opt_lin_NO_trunc, tpe_opt_lin_WITH_trunc
from experiments.synthetic.dynamic.tobit_based.reparam_fixed_std.reparametrized_gll_synthetic_heter_data import tpe_opt_gll_reparam
from experiments.synthetic.dynamic.tobit_based.reparam_fixed_std.reparametrized_tobit_loss_synthetic_heter_data import tpe_opt_deep_NO_trunc_reparam, \
    tpe_opt_deep_WITH_trunc_reparam, tpe_opt_lin_NO_trunc_reparam, tpe_opt_lin_WITH_trunc_reparam
from experiments.synthetic.dynamic.tobit_based.scaled_dynamic_std.scaled_tobit_loss_dynamic_std_synthetic_heter_data import tpe_opt_deep_WITH_trunc_dyn_std
from experiments.synthetic.dynamic.tobit_based.reparam_dynamic_std.reparametrized_tobit_loss_dynamic_std_synthetic_heter_data import tpe_opt_deep_WITH_trunc_reparam_dyn_std
from experiments.util import TruncatedBetaDistributionConfig
from experiments.synthetic.dynamic.beta_distributions_variations import get_distirbution_variations

def train_all_losses(datast_config: TruncatedBetaDistributionConfig):
    # MAE Based
    print('-' * 20, 'tpe_opt_mae_simple', '-' * 20)
    tpe_opt_mae_simple(datast_config)
    print('-' * 20, 'tpe_opt_mae_cens_NO_trunc', '-' * 20)
    tpe_opt_mae_cens_NO_trunc(datast_config)
    print('-' * 20, 'tpe_opt_mae_cens_WITH_trunc', '-' * 20)
    tpe_opt_mae_cens_WITH_trunc(datast_config)

    # MSE Based
    print('-' * 20, 'tpe_opt_mse_simple', '-' * 20)
    tpe_opt_mse_simple(datast_config)
    print('-' * 20, 'tpe_opt_mse_cens_NO_trunc', '-' * 20)
    tpe_opt_mse_cens_NO_trunc(datast_config)
    print('-' * 20, 'tpe_opt_mse_cens_WITH_trunc', '-' * 20)
    tpe_opt_mse_cens_WITH_trunc(datast_config)

    # Tobit

    # Scaled, Fixed Std
    print('-' * 20, 'tpe_opt_gll_scaled', '-' * 20)
    tpe_opt_gll_scaled(datast_config)
    print('-' * 20, 'tpe_opt_deep_NO_trunc', '-' * 20)
    tpe_opt_deep_NO_trunc(datast_config)
    print('-' * 20, 'tpe_opt_deep_WITH_trunc', '-' * 20)
    tpe_opt_deep_WITH_trunc(datast_config)
    print('-' * 20, 'tpe_opt_lin_NO_trunc', '-' * 20)
    tpe_opt_lin_NO_trunc(datast_config)
    print('-' * 20, 'tpe_opt_lin_WITH_trunc', '-' * 20)
    tpe_opt_lin_WITH_trunc(datast_config)

    # Reparametrized
    print('-' * 20, 'tpe_opt_gll_reparam', '-' * 20)
    tpe_opt_gll_reparam(datast_config)
    print('-' * 20, 'tpe_opt_deep_NO_trunc_reparam', '-' * 20)
    tpe_opt_deep_NO_trunc_reparam(datast_config)
    print('-' * 20, 'tpe_opt_deep_WITH_trunc_reparam', '-' * 20)
    tpe_opt_deep_WITH_trunc_reparam(datast_config)
    print('-' * 20, 'tpe_opt_lin_NO_trunc_reparam', '-' * 20)
    tpe_opt_lin_NO_trunc_reparam(datast_config)
    print('-' * 20, 'tpe_opt_lin_WITH_trunc_reparam', '-' * 20)
    tpe_opt_lin_WITH_trunc_reparam(datast_config)

    # Dynamic Std Scaled
    print('-' * 20, 'tpe_opt_deep_WITH_trunc_dyn_std', '-' * 20)
    tpe_opt_deep_WITH_trunc_dyn_std(datast_config)

    # Dynamic Std Reparametrized
    print('-' * 20, 'tpe_opt_deep_WITH_trunc_reparam_dyn_std', '-' * 20)
    tpe_opt_deep_WITH_trunc_reparam_dyn_std(datast_config)

for datataset_config in get_distirbution_variations():
    print(datataset_config)
    train_all_losses(datataset_config)