import json
import torch as t
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from experiments.constants import IS_CUDA_AVILABLE
from experiments.constants import DOT_SIZE
from experiments.constants import PLOT_FONT_SIZE

def dump_json(obj, path):
    with open(path, mode='w') as file:
        json.dump(obj, file)

def read_json_file(path):
    with open(path) as file:
        return json.loads(file.read())

"""Reproducible experiments"""
def set_random_seed():
    SEED = 0
    t.manual_seed(SEED)
    t.cuda.manual_seed(SEED)
    t.cuda.manual_seed_all(SEED)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

def get_device(cuda = IS_CUDA_AVILABLE):
    return 'cuda:0' if cuda else 'cpu'

def get_scale():
    scale = t.tensor(1., requires_grad = True, device = get_device())
    return scale

def load_checkpoint(checkpoint_path):
    return t.load(checkpoint_path) if IS_CUDA_AVILABLE else t.load(checkpoint_path, map_location=t.device('cpu'))

def save_figures(file_path: str):
    plt.savefig('{}.pdf'.format(file_path), dpi = 300, format = 'pdf')
    plt.savefig('{}.eps'.format(file_path), dpi = 300, format = 'eps')

def scatterplot(x, y, label = None, s = DOT_SIZE):
    plt.scatter(x, y, s = s, label = label, rasterized=True, marker='.', linewidths=0)

def save_fig_in_checkpoint_folder(root_folder, checkpoint_name, suffix = ''):
    file_path = root_folder + '/' + checkpoint_name
    plt.savefig(f'{file_path}{suffix}.pdf', dpi = 300, format = 'pdf')
    # plt.savefig(f'{file_path}{suffix}.svg', dpi = 300, format = 'svg')
    # plt.savefig(f'{file_path}{suffix}.png', dpi = 200, format = 'png')
    plt.close()

def setup_composed_5_items_plot():
    plt.rcParams.update({'font.size': PLOT_FONT_SIZE})
    fig = plt.figure()
    fig.set_size_inches(6, 6)

    gs = gridspec.GridSpec(3, 4, figure=fig)
    gs.update(wspace=.5)
    gs.update(hspace=.5)

    ax1 = plt.subplot(gs[0, 1:3])
    ax2 = plt.subplot(gs[1, :2])
    ax3 = plt.subplot(gs[1, 2:])
    ax4 = plt.subplot(gs[2, :2])
    ax5 = plt.subplot(gs[2, 2:])

    return ax1, ax2, ax3, ax4, ax5

def setup_composed_6_items_plot():
    plt.rcParams.update({'font.size': PLOT_FONT_SIZE})
    fig = plt.figure()
    fig.set_size_inches(6, 6)

    gs = gridspec.GridSpec(3, 4, figure=fig)
    gs.update(wspace=.5)
    gs.update(hspace=.5)

    ax1 = plt.subplot(gs[0, :2])
    ax2 = plt.subplot(gs[0, 2:])
    ax3 = plt.subplot(gs[1, :2])
    ax4 = plt.subplot(gs[1, 2:])
    ax5 = plt.subplot(gs[2, :2])
    ax6 = plt.subplot(gs[2, 2:])

    return ax1, ax2, ax3, ax4, ax5, ax6

def setup_composed_4_items_plot():
    plt.rcParams.update({'font.size': PLOT_FONT_SIZE})
    fig = plt.figure()
    fig.set_size_inches(6, 6)

    gs = gridspec.GridSpec(2, 4, figure=fig)
    gs.update(wspace=.5)
    gs.update(hspace=.5)

    ax1 = plt.subplot(gs[0, :2])
    ax2 = plt.subplot(gs[0, 2:])
    ax3 = plt.subplot(gs[1, :2])
    ax4 = plt.subplot(gs[1, 2:])

    return ax1, ax2, ax3, ax4