import torch as t

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