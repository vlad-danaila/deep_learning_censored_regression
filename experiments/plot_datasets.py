import matplotlib.pyplot as plt
from experiments.util import save_figures
from experiments.constants import PLOT_FONT_SIZE
from deep_tobit.util import normalize

from experiments.real.pm25.plot import plot_full_dataset as plot_pm25
from experiments.real.pm25.dataset import df as df_pm25, test_df as test_df_pm25, train_df as train_df_pm25

from experiments.real.bike_sharing.plot import plot_full_dataset as plot_bike
from experiments.real.bike_sharing.dataset import df as df_bike, test_df as test_df_bike, train_df as train_df_bike

from experiments.synthetic.plot import plot_beta, plot_dataset as plot_dataset_synthetic
from experiments.synthetic.constant_noise.dataset import calculate_mean_std as calculate_mean_std_const_noise, \
    TruncatedBetaDistributionDataset as TruncatedBetaDistributionDataset_ConstNoise, \
    TruncatedBetaDistributionValidationDataset as TruncatedBetaDistributionValidationDataset_ConstNoise
from experiments.synthetic.constants import CENSOR_LOW_BOUND, CENSOR_HIGH_BOUND


def plot_real_datasets():
    plt.rcParams.update({'font.size': PLOT_FONT_SIZE})

    fig, axs = plt.subplots(2, 2)

    plt.axes(axs[0, 0])
    plt.title('(a) PM2.5 Training Data Set')
    plot_pm25(train_df_pm25(df_pm25), label ='training data', censored = True)
    plt.ylim(-2, 7)
    plt.xlim(-4, 5)

    plt.axes(axs[0, 1])
    plt.title('(b) PM2.5 Testing Data Set')
    plot_pm25(test_df_pm25(df_pm25), label ='testing data', show_bounds = False)
    plt.ylim(-2, 7)
    plt.xlim(-4, 5)

    plt.axes(axs[1, 0])
    plt.title('(c) Seoul Bike-Sharing Training Data Set')
    plot_bike(train_df_bike(df_bike), label ='training data', censored = True)
    plt.ylim(-2, 7)
    plt.xlim(-5, 5)

    plt.axes(axs[1, 1])
    plt.title('(d) Seoul Bike-Sharing Testing Data Set')
    plot_bike(test_df_bike(df_bike), label ='testing data', show_bounds = False)
    plt.ylim(-2, 7)
    plt.xlim(-5, 5)

    plt.tight_layout()

    save_figures('experiments/all_img/datasets_real')
    plt.close()

def plot_synthetic_datasets():
    plt.rcParams.update({'font.size': PLOT_FONT_SIZE})

    # cn in the names below signify constant noise
    x_mean_cn, x_std_cn, y_mean_cn, y_std_cn = calculate_mean_std_const_noise(lower_bound = CENSOR_LOW_BOUND, upper_bound = CENSOR_HIGH_BOUND)
    dataset_train_cn = TruncatedBetaDistributionDataset_ConstNoise(x_mean_cn, x_std_cn, y_mean_cn, y_std_cn, lower_bound = CENSOR_LOW_BOUND, upper_bound = CENSOR_HIGH_BOUND)
    dataset_test_cn = TruncatedBetaDistributionValidationDataset_ConstNoise(x_mean_cn, x_std_cn, y_mean_cn, y_std_cn)

    fig, axs = plt.subplots(2, 2)

    plt.axes(axs[0, 0])
    plt.title('(a) Constant Noise Training Data Set')
    plot_dataset_synthetic(dataset_train_cn, label ='training data')
    plot_beta(x_mean_cn, x_std_cn, y_mean_cn, y_std_cn, label = 'true distribution')
    plt.ylim(-3, 3)
    lgnd = plt.legend()
    lgnd.legendHandles[0]._sizes = [10]
    lgnd.legendHandles[1]._sizes = [10]

    plt.axes(axs[0, 1])
    plt.title('(b) Constant Noise Testing Data Set')
    plot_dataset_synthetic(dataset_test_cn, label ='testing data')
    plot_beta(x_mean_cn, x_std_cn, y_mean_cn, y_std_cn, label = 'true distribution')
    plt.ylim(-3, 3)
    lgnd = plt.legend()
    lgnd.legendHandles[0]._sizes = [10]
    lgnd.legendHandles[1]._sizes = [10]

    plt.axes(axs[1, 0])
    plt.title('(c) Heteroscedastic Training Data Set')
    plot_bike(train_df_bike(df_bike), label ='training data', censored = True)
    plt.ylim(-2, 7)
    plt.xlim(-5, 5)


    plt.axes(axs[1, 1])
    plt.title('(d) Heteroscedastic Testing Data Set')
    plot_bike(test_df_bike(df_bike), label ='testing data', show_bounds = False)
    plt.ylim(-2, 7)
    plt.xlim(-5, 5)

    plt.tight_layout()

    save_figures('experiments/all_img/datasets_synthetic')
    plt.close()

plot_synthetic_datasets()

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