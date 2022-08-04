import matplotlib.pyplot as plt
from experiments.real.pm25.plot import plot_full_dataset as plot_pm25
from experiments.real.pm25.dataset import df, test_df, train_df

# fig, axs = plt.subplots(1, 2)

plot_pm25(test_df(df), size = .3, label = 'testing data', show_bounds = False)
plt.ylim(-2, 7)
plt.show()


plot_pm25(train_df(df), size = .3, label = 'training data', censored = True)
plt.ylim(-2, 7)
plt.show()