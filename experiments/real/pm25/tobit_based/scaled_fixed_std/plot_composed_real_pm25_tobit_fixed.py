import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from experiments.util import save_figures, setup_composed_5_items_plot


from experiments.real.pm25.tobit_based.scaled_fixed_std.scaled_tobit_pm25 import \
    plot_deep_tobit_NO_trunc, plot_deep_tobit_WITH_trunc, plot_linear_tobit_NO_trunc, plot_linear_tobit_WITH_trunc
from experiments.real.pm25.tobit_based.scaled_fixed_std.scaled_gll_pm25 import plot_gll_scaled

def plot_composed_pm25_tobit_fixed_std():
    ax1, ax2, ax3, ax4, ax5 = setup_composed_5_items_plot()




plot_composed_pm25_tobit_fixed_std()