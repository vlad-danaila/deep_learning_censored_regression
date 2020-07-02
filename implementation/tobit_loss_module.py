from implementation.util import to_numpy, to_torch
import torch as t
from scipy.stats import norm
import numpy as np
from implementation.util import normalize

class TobitLoss(t.nn.Module):

    def __init__(self):
        super(TobitLoss, self).__init__()

    def forward(self, single_value, left_censored, right_censored):
        pass