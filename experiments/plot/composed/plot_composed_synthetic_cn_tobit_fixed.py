import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from experiments.util import save_figures
from experiments.constants import PLOT_FONT_SIZE

from experiments.synthetic.constant_noise.tobit_based.scaled_fixed_std.scaled_tobit_loss_synthetic_data import \
    plot_deep_tobit_WITH_trunc, plot_deep_tobit_NO_trunc, plot_linear_tobit_WITH_trunc, plot_linear_tobit_NO_trunc
from experiments.synthetic.constant_noise.tobit_based.scaled_fixed_std.scaled_gll_synthetic_data import plot_gll_scaled


def plot_composed_synthetic_cn_tobit_fixed_std():
    plt.rcParams.update({'font.size': PLOT_FONT_SIZE})

    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=.5)
    gs.update(hspace=.3)
    ax1 = plt.subplot(gs[0, 1:3])
    ax2 = plt.subplot(gs[1, :2], )
    ax3 = plt.subplot(gs[1, 2:])


    plt.axes(ax1)
    # plt.title('(a) PM2.5 Training Data Set')
    plot_deep_tobit_WITH_trunc()
    # plt.ylim(-2, 7)
    # plt.xlim(-4, 5)

    plt.axes(ax2)
    plot_deep_tobit_NO_trunc()

    plt.axes(ax3)
    plot_gll_scaled()

    plt.show()
    save_figures('experiments/all_img/synt_cn_tobit_fixed_std')

plot_composed_synthetic_cn_tobit_fixed_std()

# EXAMPLE FOR MERGING PLOTS ON THE SAME ROW
# import matplotlib.gridspec as gridspec
# gs = gridspec.GridSpec(2, 4)
# gs.update(wspace=0.5)
# ax1 = plt.subplot(gs[0, :2], )
# ax2 = plt.subplot(gs[0, 2:])
# ax3 = plt.subplot(gs[1, 1:3])
#
# plt.axes(ax1)
# plt.title('(a) PM2.5 Training Data Set')
# plot_pm25(train_df_pm25(df_pm25), label ='training data', censored = True)
# plt.ylim(-2, 7)
# plt.xlim(-4, 5)
#
# plt.axes(ax2)
# plt.title('(b) PM2.5 Testing Data Set')
# plot_pm25(test_df_pm25(df_pm25), label ='testing data', show_bounds = False)
# plt.ylim(-2, 7)
# plt.xlim(-4, 5)
#
# plt.axes(ax3)
# plt.title('(c) Seoul Bike-Sharing Training Data Set')
# plot_bike(train_df_bike(df_bike), label ='training data', censored = True)
# plt.ylim(-2, 7)
# plt.xlim(-5, 5)
#
# plt.show()