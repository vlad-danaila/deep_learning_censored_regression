from experiments.synthetic.constant_noise.mae_based.mae_based_synthetic_const_noise import eval_mae_simple, eval_mae_cens_NO_trunc, eval_mae_cens_WITH_trunc
from experiments.synthetic.constant_noise.mse_based.mse_based_synthetic_const_noise import eval_mse_simple, eval_mse_cens_NO_trunc, eval_mse_cens_WITH_trunc
from experiments.synthetic.constant_noise.tobit_based.scaled_fixed_std.scaled_gll_synthetic_data import tpe_opt_gll_scaled
from experiments.synthetic.constant_noise.tobit_based.scaled_fixed_std.scaled_tobit_loss_synthetic_data import tpe_opt_deep_NO_trunc, tpe_opt_deep_WITH_trunc, \
    tpe_opt_lin_NO_trunc, tpe_opt_lin_WITH_trunc
from experiments.synthetic.constant_noise.tobit_based.reparam_fixed_std.reparametrized_gll_synthetic_data import tpe_opt_gll_reparam
from experiments.synthetic.constant_noise.tobit_based.reparam_fixed_std.reparametrized_tobit_loss_synthetic_data import tpe_opt_deep_NO_trunc_reparam, \
    tpe_opt_deep_WITH_trunc_reparam, tpe_opt_lin_NO_trunc_reparam, tpe_opt_lin_WITH_trunc_reparam
from experiments.synthetic.constant_noise.tobit_based.scaled_dynamic_std.scaled_tobit_loss_dynamic_std_synthetic_const_noise_data import tpe_opt_deep_WITH_trunc_dyn_std
from experiments.synthetic.constant_noise.tobit_based.reparam_dynamic_std.reparametrized_tobit_loss_dynamic_std_synthetic_const_noise_data import tpe_opt_deep_WITH_trunc_reparam_dyn_std

# MAE Based
# eval_mae_simple()
# eval_mae_cens_NO_trunc()
# eval_mae_cens_WITH_trunc()

# MSE Based
# eval_mse_simple()
# eval_mse_cens_NO_trunc()
# eval_mse_cens_WITH_trunc()

# Tobit

# Scaled, Fixed Std
tpe_opt_gll_scaled()
tpe_opt_deep_NO_trunc()
tpe_opt_deep_WITH_trunc()
tpe_opt_lin_NO_trunc()
tpe_opt_lin_WITH_trunc()

# Reparametrized
tpe_opt_gll_reparam()
tpe_opt_deep_NO_trunc_reparam()
tpe_opt_deep_WITH_trunc_reparam()
tpe_opt_lin_NO_trunc_reparam()
tpe_opt_lin_WITH_trunc_reparam()

# Dynamic Std Scaled
tpe_opt_deep_WITH_trunc_dyn_std()

# Dynamic Std Reparametrized
tpe_opt_deep_WITH_trunc_reparam_dyn_std()