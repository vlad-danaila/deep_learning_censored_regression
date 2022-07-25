from experiments.synthetic.constant_noise.mae_based.mae_based_synthetic_const_noise import eval_mae_simple, eval_mae_cens_NO_trunc, eval_mae_cens_WITH_trunc
from experiments.synthetic.constant_noise.mse_based.mse_based_synthetic_const_noise import eval_mse_simple, eval_mse_cens_NO_trunc, eval_mse_cens_WITH_trunc
from experiments.synthetic.constant_noise.tobit_based.scaled_fixed_std.scaled_gll_synthetic_data import eval_gll_scaled

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

# Reparametrized