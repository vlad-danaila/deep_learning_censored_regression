import matplotlib.pyplot as plt
from experiments.util import save_figures, setup_composed_4_items_plot

from experiments.synthetic.constant_noise.tobit_based.scaled_fixed_std.scaled_tobit_loss_synthetic_data import plot_deep_tobit_WITH_trunc
from experiments.synthetic.constant_noise.tobit_based.scaled_dynamic_std.scaled_tobit_loss_dynamic_std_synthetic_const_noise_data import plot_deep_tobit_WITH_trunc_dyn_std

from experiments.synthetic.constant_noise.tobit_based.reparam_fixed_std.reparametrized_tobit_loss_synthetic_data import plot_deep_tobit_WITH_trunc_reparam
from experiments.synthetic.constant_noise.tobit_based.reparam_dynamic_std.reparametrized_tobit_loss_dynamic_std_synthetic_const_noise_data import plot_deep_tobit_WITH_trunc_reparam_dyn_std

def plot_composed_synthetic_cn_dyn_std():
    ax1, ax2, ax3, ax4 = setup_composed_4_items_plot()

    plt.axes(ax1)
    plot_deep_tobit_WITH_trunc()
    plt.title('(a) Fixed Standard Deviation')

    plt.axes(ax2)
    plot_deep_tobit_WITH_trunc_dyn_std()
    plt.title('(b) Dynamic Standard Deviation')

    plt.axes(ax3)
    plot_deep_tobit_WITH_trunc_reparam()
    plt.title('(c) Fixed Standard Deviation Reparametrized')

    plt.axes(ax4)
    plot_deep_tobit_WITH_trunc_reparam_dyn_std()
    plt.title('(d) Dynamic Standard Deviation Reparametrized')

    plt.show()
    save_figures('experiments/all_img/synt_cn_dyn_std_compare')


plot_composed_synthetic_cn_dyn_std()