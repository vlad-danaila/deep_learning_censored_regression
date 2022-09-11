from experiments.real.bike_sharing.mae_based.mae_based_bike import eval_mae_simple, eval_mae_cens_NO_trunc, eval_mae_cens_WITH_trunc
from experiments.real.bike_sharing.mse_based.mse_based_bike import eval_mse_simple, eval_mse_cens_NO_trunc, eval_mse_cens_WITH_trunc
from experiments.real.bike_sharing.tobit_based.scaled_fixed_std.scaled_gll_bike import tpe_opt_gll_scaled
from experiments.real.bike_sharing.tobit_based.scaled_fixed_std.scaled_tobit_bike import tpe_opt_deep_NO_trunc, tpe_opt_deep_WITH_trunc, \
    tpe_opt_lin_NO_trunc, tpe_opt_lin_WITH_trunc
from experiments.real.bike_sharing.tobit_based.reparam_fixed_std.reparam_gll_bike import tpe_opt_gll_reparam
from experiments.real.bike_sharing.tobit_based.reparam_fixed_std.reparametrized_tobit_bike import tpe_opt_deep_NO_trunc_reparam, \
    tpe_opt_deep_WITH_trunc_reparam, tpe_opt_lin_NO_trunc_reparam, tpe_opt_lin_WITH_trunc_reparam
from experiments.real.bike_sharing.tobit_based.scaled_dynamic_std.scaled_tobit_loss_dynamic_std_bike import tpe_opt_deep_WITH_trunc_dyn_std
from experiments.real.bike_sharing.tobit_based.reparam_dynamic_std.reparametrized_tobit_loss_dynamic_std_bike import tpe_opt_deep_WITH_trunc_dyn_std_reparam

# MAE Based
eval_mae_simple()
eval_mae_cens_NO_trunc()
eval_mae_cens_WITH_trunc()

# MSE Based
eval_mse_simple()
eval_mse_cens_NO_trunc()
eval_mse_cens_WITH_trunc()

# Tobit

# Scaled, Fixed Std
print('-' * 20, 'tpe_opt_gll_scaled', '-' * 20)
tpe_opt_gll_scaled()
print('-' * 20, 'tpe_opt_deep_NO_trunc', '-' * 20)
tpe_opt_deep_NO_trunc()
print('-' * 20, 'tpe_opt_deep_WITH_trunc', '-' * 20)
tpe_opt_deep_WITH_trunc()
print('-' * 20, 'tpe_opt_lin_NO_trunc', '-' * 20)
tpe_opt_lin_NO_trunc()
print('-' * 20, 'tpe_opt_lin_WITH_trunc', '-' * 20)
tpe_opt_lin_WITH_trunc()

# Reparametrized
print('-' * 20, 'tpe_opt_gll_reparam', '-' * 20)
tpe_opt_gll_reparam()
print('-' * 20, 'tpe_opt_deep_NO_trunc_reparam', '-' * 20)
tpe_opt_deep_NO_trunc_reparam()
print('-' * 20, 'tpe_opt_deep_WITH_trunc_reparam', '-' * 20)
tpe_opt_deep_WITH_trunc_reparam()
print('-' * 20, 'tpe_opt_lin_NO_trunc_reparam', '-' * 20)
tpe_opt_lin_NO_trunc_reparam()
print('-' * 20, 'tpe_opt_lin_WITH_trunc_reparam', '-' * 20)
tpe_opt_lin_WITH_trunc_reparam()

# Dynamic Std Scaled
print('-' * 20, 'tpe_opt_deep_WITH_trunc_dyn_std', '-' * 20)
tpe_opt_deep_WITH_trunc_dyn_std()

# Dynamic Std Reparametrized
print('-' * 20, 'tpe_opt_deep_WITH_trunc_reparam_dyn_std', '-' * 20)
tpe_opt_deep_WITH_trunc_dyn_std_reparam()