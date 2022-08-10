import matplotlib.pyplot as plt
from experiments.util import save_figures, setup_composed_5_items_plot

from experiments.synthetic.constant_noise.tobit_based.reparam_fixed_std.reparametrized_tobit_loss_synthetic_data import \
    plot_deep_tobit_WITH_trunc_reparam, plot_deep_tobit_NO_trunc_reparam, plot_linear_tobit_WITH_trunc_reparam, plot_linear_tobit_NO_trunc_reparam
from experiments.synthetic.constant_noise.tobit_based.reparam_fixed_std.reparametrized_gll_synthetic_data import plot_gll_reparam


def plot_composed_synthetic_cn_reparam_tobit_fixed_std():
    ax1, ax2, ax3, ax4, ax5 = setup_composed_5_items_plot()

    plt.axes(ax1)
    plot_deep_tobit_WITH_trunc_reparam()
    plt.title('(a) Tobit Log-Likelihood With Truncation')

    plt.axes(ax2)
    plot_deep_tobit_NO_trunc_reparam()
    plt.title('(b) Tobit Log-Likelihood (No Truncation)')

    plt.axes(ax3)
    plot_gll_reparam()
    plt.title('(c) Log-Likelihood (No Censoring, No Truncation)')

    plt.axes(ax4)
    plot_linear_tobit_WITH_trunc_reparam()
    plt.title('(d) Linear Tobit Log-Likelihood With Truncation')

    plt.axes(ax5)
    plot_linear_tobit_NO_trunc_reparam()
    plt.title('(e) Linear Tobit Log-Likelihood (No Truncation)')

    plt.show()
    save_figures('experiments/all_img/synt_cn_reparam_tobit_fixed_std')


plot_composed_synthetic_cn_reparam_tobit_fixed_std()