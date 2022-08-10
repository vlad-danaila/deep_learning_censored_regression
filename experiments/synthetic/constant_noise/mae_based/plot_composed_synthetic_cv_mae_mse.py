import matplotlib.pyplot as plt
from experiments.util import save_figures, setup_composed_6_items_plot

from experiments.synthetic.constant_noise.mae_based.mae_based_synthetic_const_noise import plot_mae_simple, plot_mae_cens_NO_trunc, plot_mae_cens_WITH_trunc
from experiments.synthetic.constant_noise.mse_based.mse_based_synthetic_const_noise import plot_mse_simple, plot_mse_cens_NO_trunc, plot_mse_cens_WITH_trunc


def plot_composed_synthetic_cn_mae_mse():
    ax1, ax2, ax3, ax4, ax5, ax6 = setup_composed_6_items_plot()

    plt.axes(ax1)
    plot_mae_cens_WITH_trunc()
    plt.title('(a) Censored Mean Absolute Error With Truncation')

    plt.axes(ax2)
    plot_mae_cens_NO_trunc()
    plt.title('(b) Censored Mean Absolute Error (No Truncation)')

    plt.axes(ax3)
    plot_mae_simple()
    plt.title('(c) Mean Absolute Error (No Truncation, No Censoring)')

    plt.axes(ax4)
    plot_mse_cens_WITH_trunc()
    plt.title('(d) Censored Mean Squared Error With Truncation')

    plt.axes(ax5)
    plot_mse_cens_NO_trunc()
    plt.title('(e) Censored Mean Squared Error (No Truncation)')

    plt.axes(ax6)
    plot_mse_simple()
    plt.title('(f) Mean Squared Error (No Truncation, No Censoring)')

    plt.show()
    save_figures('experiments/all_img/synt_cn_mae_mse')


plot_composed_synthetic_cn_mae_mse()