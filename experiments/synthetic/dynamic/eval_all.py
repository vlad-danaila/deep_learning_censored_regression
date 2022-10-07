from experiments.synthetic.dynamic.mae_based.mae_based_synthetic_heter import eval_mae_simple, eval_mae_cens_NO_trunc, eval_mae_cens_WITH_trunc
from experiments.synthetic.dynamic.mse_based.mse_based_synthetic_heter import eval_mse_simple, eval_mse_cens_NO_trunc, eval_mse_cens_WITH_trunc
from experiments.synthetic.dynamic.tobit_based.scaled_fixed_std.scaled_gll_synthetic_heter_data import eval_gll_scaled
from experiments.synthetic.dynamic.tobit_based.scaled_fixed_std.scaled_tobit_loss_synthetic_heter_data import eval_deep_NO_trunc, eval_deep_WITH_trunc,\
    eval_lin_NO_trunc, eval_linear_tobit_WITH_trunc
from experiments.synthetic.dynamic.tobit_based.reparam_fixed_std.reparametrized_gll_synthetic_heter_data import eval_gll_reparam
from experiments.synthetic.dynamic.tobit_based.reparam_fixed_std.reparametrized_tobit_loss_synthetic_heter_data import eval_deep_NO_trunc_reparam,\
    eval_deep_WITH_trunc_reparam, eval_linear_tobit_NO_trunc_reparam, eval_linear_tobit_WITH_trunc_reparam
from experiments.synthetic.dynamic.tobit_based.scaled_dynamic_std.scaled_tobit_loss_dynamic_std_synthetic_heter_data import eval_deep_WITH_trunc_dyn_std
from experiments.synthetic.dynamic.tobit_based.reparam_dynamic_std.reparametrized_tobit_loss_dynamic_std_synthetic_heter_data import eval_deep_WITH_trunc_reparam_dyn_std
from experiments.util import TruncatedBetaDistributionConfig
from experiments.synthetic.dynamic.beta_distributions_variations import get_distirbution_variations

def eval_all_losses(datast_config: TruncatedBetaDistributionConfig):
    # MAE Based
    eval_mae_simple(datast_config)
    eval_mae_cens_NO_trunc(datast_config)
    eval_mae_cens_WITH_trunc(datast_config)

    # MSE Based
    eval_mse_simple(datast_config)
    eval_mse_cens_NO_trunc(datast_config)
    eval_mse_cens_WITH_trunc(datast_config)

    # Tobit

    # Scaled, Fixed Std
    eval_gll_scaled(datast_config)
    eval_deep_NO_trunc(datast_config)
    eval_deep_WITH_trunc(datast_config)
    eval_lin_NO_trunc(datast_config)
    eval_linear_tobit_WITH_trunc(datast_config)

    # Reparametrized
    eval_gll_reparam(datast_config)
    eval_deep_NO_trunc_reparam(datast_config)
    eval_deep_WITH_trunc_reparam(datast_config)
    eval_linear_tobit_NO_trunc_reparam(datast_config)
    eval_linear_tobit_WITH_trunc_reparam(datast_config)

    # Dynamic Std Scaled
    eval_deep_WITH_trunc_dyn_std(datast_config)

    # Dynamic Std Reparametrized
    eval_deep_WITH_trunc_reparam_dyn_std(datast_config)

for datataset_config in get_distirbution_variations():
    print(datataset_config)
    eval_all_losses(datataset_config)