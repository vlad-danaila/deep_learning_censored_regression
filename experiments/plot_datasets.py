import matplotlib.pyplot as plt
from experiments.real.pm25.plot import plot_full_dataset as plot_pm25
from experiments.real.pm25.dataset import df as df_pm25, test_df as test_df_pm25, train_df as train_df_pm25
from experiments.real.bike_sharing.plot import plot_full_dataset as plot_bike
from experiments.real.bike_sharing.dataset import df as df_bike, test_df as test_df_bike, train_df as train_df_bike
from experiments.util import save_figures


plt.rcParams.update({'font.size': 5})

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

save_figures('experiments/real/real_datasets_img/real_datasets')
# plt.close()