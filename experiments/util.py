import torch as t
import numpy as np
import random
import matplotlib.pyplot as plt
from experiments.constants import IS_CUDA_AVILABLE
from experiments.constants import DOT_SIZE

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
    plt.savefig('{}.pdf'.format(file_path), dpi = 1000, format = 'pdf')
    plt.savefig('{}.eps'.format(file_path), dpi = 300, format = 'eps')

def scatterplot(x, y, label = None, s = DOT_SIZE):
    plt.scatter(x, y, s = s, label = label, rasterized=True, marker='.', linewidths=0)