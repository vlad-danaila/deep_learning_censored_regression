import matplotlib.pyplot as plt
from experiments.real.pm25.plot import plot_full_dataset as plot_pm25
from experiments.real.pm25.dataset import df, test_df, train_df
from experiments.util import save_figures

fig, axs = plt.subplots(2, 2)

plt.axes(axs[0, 0])
plt.title('Title 1')
plot_pm25(train_df(df), label = 'training data', censored = True)
plt.ylim(-2, 7)

plt.axes(axs[0, 1])
plt.title('Title 2')
plot_pm25(test_df(df), label = 'testing data', show_bounds = False)
plt.ylim(-2, 7)

plt.axes(axs[1, 0])
plt.title('Title 3')
plot_pm25(train_df(df), label = 'training data', censored = True)
plt.ylim(-2, 7)


plt.axes(axs[1, 1])
plt.title('Title 4')
plot_pm25(test_df(df), label = 'testing data', show_bounds = False)
plt.ylim(-2, 7)

plt.tight_layout()

# save_figures('experiments/real/real_datasets_img/real_datasets')
# plt.close()