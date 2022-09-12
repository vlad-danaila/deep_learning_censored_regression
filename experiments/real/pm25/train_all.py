from experiments.real.pm25.mae_based.mae_based_pm25 import tpe_opt_mae_simple, tpe_opt_mae_cens_NO_trunc, tpe_opt_mae_cens_WITH_trunc
from experiments.real.pm25.mse_based.mse_based_pm25 import tpe_opt_mse_simple, tpe_opt_mse_cens_NO_trunc, tpe_opt_mse_cens_WITH_trunc
from experiments.real.pm25.tobit_based.scaled_fixed_std.scaled_gll_pm25 import tpe_opt_gll_scaled
from experiments.real.pm25.tobit_based.scaled_fixed_std.scaled_tobit_pm25 import tpe_opt_deep_NO_trunc, tpe_opt_deep_WITH_trunc, \
    tpe_opt_lin_NO_trunc, tpe_opt_lin_WITH_trunc
from experiments.real.pm25.tobit_based.reparam_fixed_std.reparam_gll_pm25 import tpe_opt_gll_reparam
from experiments.real.pm25.tobit_based.reparam_fixed_std.reparametrized_tobit_pm25 import tpe_opt_deep_NO_trunc_reparam, \
    tpe_opt_deep_WITH_trunc_reparam, tpe_opt_lin_NO_trunc_reparam, tpe_opt_lin_WITH_trunc_reparam
from experiments.real.pm25.tobit_based.scaled_dynamic_std.scaled_tobit_loss_dynamic_std_pm25 import tpe_opt_deep_WITH_trunc_dyn_std
from experiments.real.pm25.tobit_based.reparam_dynamic_std.reparametrized_tobit_loss_dynamic_std_pm25 import tpe_opt_deep_WITH_trunc_dyn_std_reparam

# MAE Based
tpe_opt_mae_simple()
tpe_opt_mae_cens_NO_trunc()
tpe_opt_mae_cens_WITH_trunc()

# MSE Based
tpe_opt_mse_simple()
tpe_opt_mse_cens_NO_trunc()
tpe_opt_mse_cens_WITH_trunc()

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