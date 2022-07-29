import torch as t

LAYER_SIZE = 10

"""# Model"""

class DenseNetwork(t.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_in = t.nn.Linear(1, LAYER_SIZE)
        self.norm_1 = t.nn.BatchNorm1d(LAYER_SIZE, affine = False)
        self.layer_hidden_1 = t.nn.Linear(LAYER_SIZE, LAYER_SIZE)
        self.norm_2 = t.nn.BatchNorm1d(LAYER_SIZE, affine = False)
        self.layer_out = t.nn.Linear(LAYER_SIZE, 1)

    def forward(self, x):
        x = t.nn.functional.relu(self.norm_1(self.layer_in(x)))
        x = t.nn.functional.relu(self.norm_2(self.layer_hidden_1(x)))
        x = self.layer_out(x)
        return x

class ScaleNetwork(t.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_in = t.nn.Linear(1, LAYER_SIZE)
        self.norm_1 = t.nn.BatchNorm1d(LAYER_SIZE, affine = False)
        self.layer_out = t.nn.Linear(LAYER_SIZE, 1)

    def forward(self, x):
        x = self.layer_in(x)
        x = self.norm_1(x)
        x = t.nn.functional.relu(x)
        x = self.layer_out(x)
        return x

def get_scale_network():
    scale_net = ScaleNetwork()
    return scale_net