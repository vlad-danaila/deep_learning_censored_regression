import matplotlib.pyplot as plt
from experiments.util import save_figures, setup_composed_5_items_plot

from experiments.synthetic.constant_noise.tobit_based.scaled_fixed_std.scaled_tobit_loss_synthetic_data import \
    plot_deep_WITH_trunc, plot_deep_NO_trunc, plot_linear_tobit_WITH_trunc, plot_lin_NO_trunc
from experiments.synthetic.constant_noise.tobit_based.scaled_fixed_std.scaled_gll_synthetic_data import plot_gll_scaled


def plot_composed_synthetic_cn_tobit_fixed_std():
    ax1, ax2, ax3, ax4, ax5 = setup_composed_5_items_plot()

    plt.axes(ax1)
    plot_deep_WITH_trunc()
    plt.title('(a) Tobit Log-Likelihood With Truncation')

    plt.axes(ax2)
    plot_deep_NO_trunc()
    plt.title('(b) Tobit Log-Likelihood (No Truncation)')

    plt.axes(ax3)
    plot_gll_scaled()
    plt.title('(c) Log-Likelihood (No Censoring, No Truncation)')

    plt.axes(ax4)
    plot_linear_tobit_WITH_trunc()
    plt.title('(d) Linear Tobit Log-Likelihood With Truncation')

    plt.axes(ax5)
    plot_lin_NO_trunc()
    plt.title('(e) Linear Tobit Log-Likelihood (No Truncation)')

    plt.show()
    save_figures('experiments/all_img/synt_cn_tobit_fixed_std')


plot_composed_synthetic_cn_tobit_fixed_std()