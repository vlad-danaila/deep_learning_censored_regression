import matplotlib.pyplot as plt

from experiments.synthetic.constant_noise.tobit_based.scaled_fixed_std.scaled_tobit_loss_synthetic_data import \
    plot_deep_tobit_WITH_trunc, plot_deep_tobit_NO_trunc, plot_linear_tobit_WITH_trunc, plot_linear_tobit_NO_trunc
from experiments.synthetic.constant_noise.tobit_based.scaled_fixed_std.scaled_gll_synthetic_data import plot_gll_scaled


def plot_composed_synthetic_cn_tobit_fixed_std():
    plot_deep_tobit_WITH_trunc()
    plt.show()


plot_composed_synthetic_cn_tobit_fixed_std()