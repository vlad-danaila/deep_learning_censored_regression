import matplotlib.pyplot as plt
from experiments.util import save_figures
from experiments.constants import PLOT_FONT_SIZE
from deep_tobit.util import normalize

from experiments.real.pm25.plot import plot_full_dataset as plot_pm25
from experiments.real.pm25.dataset import df as df_pm25, test_df as test_df_pm25, train_df as train_df_pm25

def plot_model_eval_example():
    plt.rcParams.update({'font.size': PLOT_FONT_SIZE})
    plt.title('(a) PM2.5 Training Data Set')
    plot_pm25(train_df_pm25(df_pm25), label ='training data', censored = False)
    plt.ylim(-2, 7)
    # plt.xlim(-4, 5)
    save_figures('experiments/all_img/model_eval_example')

plot_model_eval_example()