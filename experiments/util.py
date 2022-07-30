import torch as t
import numpy as np
import random

from experiments.constants import IS_CUDA_AVILABLE

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