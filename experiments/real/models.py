import torch as t
from experiments.constants import IS_CUDA_AVILABLE

LAYER_SIZE = 10

"""# Model"""

class DenseNetwork(t.nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.layer_in = t.nn.Linear(input_size, LAYER_SIZE)
        self.norm_1 = t.nn.BatchNorm1d(LAYER_SIZE, affine = False, momentum = None)
        self.drop_1 = t.nn.Dropout(p = .2)
        self.layer_hidden_1 = t.nn.Linear(LAYER_SIZE, LAYER_SIZE)
        self.norm_2 = t.nn.BatchNorm1d(LAYER_SIZE, affine = False, momentum = None)
        self.drop_2 = t.nn.Dropout(p = .2)
        self.layer_out = t.nn.Linear(LAYER_SIZE, 1)

    def forward(self, x):
        x = t.nn.functional.relu(self.norm_1(self.layer_in(x)))
        x = self.drop_1(x)
        x = t.nn.functional.relu(self.norm_2(self.layer_hidden_1(x)))
        x = self.drop_2(x)
        x = self.layer_out(x)
        return x

def get_model(input_size, cuda = IS_CUDA_AVILABLE, net = None):
    if net == None:
        net = DenseNetwork(input_size)
    if cuda:
        net = net.cuda()
    net = t.nn.DataParallel(net)
    return net


class ScaleNetwork(t.nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.layer_in = t.nn.Linear(input_size, LAYER_SIZE)
        self.norm_1 = t.nn.BatchNorm1d(LAYER_SIZE, affine = False)
        self.layer_out = t.nn.Linear(LAYER_SIZE, 1)

    def forward(self, x):
        x = self.layer_in(x)
        x = self.norm_1(x)
        x = t.nn.functional.relu(x)
        x = self.layer_out(x)
        return x

def get_scale_network(layer_size, cuda = IS_CUDA_AVILABLE):
    scale_net = ScaleNetwork(layer_size)
    if cuda:
        scale_net = scale_net.cuda()
    return scale_net




def linear_model(input_size):
    model = t.nn.Linear(input_size, 1)
    if IS_CUDA_AVILABLE:
        model = model.cuda()
    return model