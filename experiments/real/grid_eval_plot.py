from matplotlib import pyplot as plt

from experiments.real.pm25.plot import plot_full_dataset, plot_net


def plot_dataset_and_net(checkpoint, model, testing_df, with_std=False, scale_model=None):
    model.load_state_dict(checkpoint['model'])
    plot_full_dataset(testing_df, label = 'ground truth')
    if 'gamma' in checkpoint:
        if scale_model:
            plot_net(model, testing_df, gamma_model = scale_model, with_std=with_std)
        else:
            plot_net(model, testing_df, gamma = checkpoint['gamma'], with_std=with_std)
    elif 'sigma' in checkpoint:
        if scale_model:
            plot_net(model, testing_df, sigma_model = scale_model, with_std=with_std)
        else:
            plot_net(model, testing_df, sigma = checkpoint['sigma'], with_std=with_std)
    else:
        plot_net(model, testing_df)
    plt.xlabel('unidimensional PCA')
    plt.ylabel('PM2.5 (standardized)')
    plt.ylim([-6, 9])
    lgnd = plt.legend(loc='upper left')
    lgnd.legendHandles[0]._sizes = [10]
    lgnd.legendHandles[1]._sizes = [10]
    if with_std:
        lgnd.legendHandles[2]._sizes = [10]