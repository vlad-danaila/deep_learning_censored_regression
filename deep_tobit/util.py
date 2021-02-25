import torch as t
import math

def normalize(x, mean, std, epsilon = 0):
    if x is None:
        return None
    return (x - mean) / (std + epsilon)

def unnormalize(x, mean, std):
    if x is None:
        return None
    return (x * std) + mean

def to_numpy(tensor: t.Tensor):
    return tensor.detach().cpu().numpy()

def to_torch(x, type = t.float64, device = 'cpu', grad = False):
    return t.tensor(x, dtype = type, device = device, requires_grad = grad)

def distinguish_censored_versus_observed_data(lower_bound = -math.inf, upper_bound = math.inf):
    def collate_function(batch):
        x, y = zip(*batch)
        x = t.stack(x)
        y = t.stack(y)
        single_valued_indexes = ((y > lower_bound) & (y < upper_bound)).squeeze()
        left_censored_indexes = (y <= lower_bound).squeeze()
        right_censored_indexes = (y >= upper_bound).squeeze()
        return x, y, single_valued_indexes, left_censored_indexes, right_censored_indexes

    return collate_function