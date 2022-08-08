import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from experiments.util import save_figures
from experiments.constants import PLOT_FONT_SIZE

from experiments.synthetic.constant_noise.tobit_based.scaled_fixed_std.scaled_tobit_loss_synthetic_data import \
    plot_deep_tobit_WITH_trunc, plot_deep_tobit_NO_trunc, plot_linear_tobit_WITH_trunc, plot_linear_tobit_NO_trunc
from experiments.synthetic.constant_noise.tobit_based.scaled_fixed_std.scaled_gll_synthetic_data import plot_gll_scaled


def plot_composed_synthetic_cn_tobit_fixed_std():
    plt.rcParams.update({'font.size': PLOT_FONT_SIZE})
    fig = plt.figure()
    fig.set_size_inches(6, 6)

    gs = gridspec.GridSpec(3, 4, figure=fig)
    gs.update(wspace=.5)
    gs.update(hspace=.5)

    ax1 = plt.subplot(gs[0, 1:3])
    ax2 = plt.subplot(gs[1, :2])
    ax3 = plt.subplot(gs[1, 2:])
    ax4 = plt.subplot(gs[2, :2])
    ax5 = plt.subplot(gs[2, 2:])

    plt.axes(ax1)
    plot_deep_tobit_WITH_trunc()
    plt.title('(a) Tobit Log-Likelihood With Truncation')

    plt.axes(ax2)
    plot_deep_tobit_NO_trunc()
    plt.title('(b) Tobit Log-Likelihood (No Truncation)')

    plt.axes(ax3)
    plot_gll_scaled()
    plt.title('(c) Log-Likelihood (No Censoring, No Truncation)')

    plt.axes(ax4)
    plot_linear_tobit_WITH_trunc()
    plt.title('(d) Linear Tobit Log-Likelihood With Truncation')

    plt.axes(ax5)
    plot_linear_tobit_NO_trunc()
    plt.title('(e) Linear Tobit Log-Likelihood (No Truncation)')

    plt.show()
    save_figures('experiments/all_img/synt_cn_tobit_fixed_std')


plot_composed_synthetic_cn_tobit_fixed_std()