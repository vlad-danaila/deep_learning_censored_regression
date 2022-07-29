import torch as t
import numpy as np
import random

CUDA = t.cuda.is_available()

LOSS = 0
ABS_ERR = 1
R_SQUARED = 2

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