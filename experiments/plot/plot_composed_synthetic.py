import matplotlib.pyplot as plt
from experiments.util import save_figures
from experiments.constants import PLOT_FONT_SIZE
from deep_tobit.util import normalize

from experiments.real.pm25.plot import plot_full_dataset as plot_pm25
from experiments.real.pm25.dataset import df as df_pm25, test_df as test_df_pm25, train_df as train_df_pm25

from experiments.real.bike_sharing.plot import plot_full_dataset as plot_bike
from experiments.real.bike_sharing.dataset import df as df_bike, test_df as test_df_bike, train_df as train_df_bike

from experiments.synthetic.plot import plot_beta, plot_dataset as plot_dataset_synthetic
from experiments.synthetic.constants import CENSOR_LOW_BOUND, CENSOR_HIGH_BOUND
from experiments.synthetic.constant_noise.dataset import calculate_mean_std as calculate_mean_std_const_noise, \
    TruncatedBetaDistributionDataset as TruncatedBetaDistributionDataset_ConstNoise, \
    TruncatedBetaDistributionValidationDataset as TruncatedBetaDistributionValidationDataset_ConstNoise
from experiments.synthetic.heteroscedastic.dataset import calculate_mean_std as calculate_mean_std_heter, \
    TruncatedBetaDistributionDataset as TruncatedBetaDistributionDataset_Heter, \
    TruncatedBetaDistributionValidationDataset as TruncatedBetaDistributionValidationDataset_Heter

def plot_composed_synthetic():
    pass