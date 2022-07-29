import torch as t
from experiments.constants import CUDA

LAYER_SIZE = 10

"""# Model"""

class DenseNetwork(t.nn.Module):

    def __init__(self, layer_size):
        super().__init__()
        self.layer_in = t.nn.Linear(46, layer_size)
        self.norm_1 = t.nn.BatchNorm1d(layer_size, affine = False, momentum = None)
        self.drop_1 = t.nn.Dropout(p = .2)
        self.layer_hidden_1 = t.nn.Linear(layer_size, layer_size)
        self.norm_2 = t.nn.BatchNorm1d(layer_size, affine = False, momentum = None)
        self.drop_2 = t.nn.Dropout(p = .2)
        self.layer_out = t.nn.Linear(layer_size, 1)

    def forward(self, x):
        x = t.nn.functional.relu(self.norm_1(self.layer_in(x)))
        x = self.drop_1(x)
        x = t.nn.functional.relu(self.norm_2(self.layer_hidden_1(x)))
        x = self.drop_2(x)
        x = self.layer_out(x)
        return x

def get_model(cuda = CUDA, net = None):
    if net == None:
        net = DenseNetwork()
    if cuda:
        net = net.cuda()
    net = t.nn.DataParallel(net)
    return net

def get_device(cuda = CUDA):
    return 'cuda:0' if cuda else 'cpu'

def get_scale(cuda = CUDA):
    scale = t.tensor(1., requires_grad = True, device = get_device())
    return scale

class ScaleNetwork(t.nn.Module):

    def __init__(self, layer_size):
        super().__init__()
        self.layer_in = t.nn.Linear(1, layer_size)
        self.norm_1 = t.nn.BatchNorm1d(layer_size, affine = False)
        self.layer_out = t.nn.Linear(layer_size, 1)

    def forward(self, x):
        x = self.layer_in(x)
        x = self.norm_1(x)
        x = t.nn.functional.relu(x)
        x = self.layer_out(x)
        return x

def get_scale_network(cuda = CUDA):
    scale_net = ScaleNetwork()
    if cuda:
        scale_net = scale_net.cuda()
    return scale_net