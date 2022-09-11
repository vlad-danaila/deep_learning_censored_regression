from experiments.real.bike_sharing.mae_based.mae_based_bike import eval_mae_simple, eval_mae_cens_NO_trunc, eval_mae_cens_WITH_trunc
from experiments.real.bike_sharing.mse_based.mse_based_bike import eval_mse_simple, eval_mse_cens_NO_trunc, eval_mse_cens_WITH_trunc
from experiments.real.bike_sharing.tobit_based.scaled_fixed_std.scaled_gll_bike import eval_gll_scaled
from experiments.real.bike_sharing.tobit_based.scaled_fixed_std.scaled_tobit_bike import eval_deep_NO_trunc, eval_deep_WITH_trunc,\
    eval_linear_NO_trunc, eval_linear_tobit_WITH_trunc
from experiments.real.bike_sharing.tobit_based.reparam_fixed_std.reparam_gll_bike import eval_gll_reparam
from experiments.real.bike_sharing.tobit_based.reparam_fixed_std.reparametrized_tobit_bike import eval_deep_NO_trunc_reparam,\
    eval_deep_WITH_trunc_reparam, eval_linear_NO_trunc_reparam, eval_linear_tobit_WITH_trunc_reparam
from experiments.real.bike_sharing.tobit_based.scaled_dynamic_std.scaled_tobit_loss_dynamic_std_bike import eval_deep_WITH_trunc_dyn_std
from experiments.real.bike_sharing.tobit_based.reparam_dynamic_std.reparametrized_tobit_loss_dynamic_std_bike import eval_deep_reparam_WITH_trunc_dyn_std

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
eval_gll_scaled()
eval_deep_NO_trunc()
eval_deep_WITH_trunc()
eval_linear_NO_trunc()
eval_linear_tobit_WITH_trunc()

# Reparametrized
eval_gll_reparam()
eval_deep_NO_trunc_reparam()
eval_deep_WITH_trunc_reparam()
eval_linear_NO_trunc_reparam()
eval_linear_tobit_WITH_trunc_reparam()

# Dynamic Std Scaled
eval_deep_WITH_trunc_dyn_std()

# Dynamic Std Reparametrized
eval_deep_reparam_WITH_trunc_dyn_std()