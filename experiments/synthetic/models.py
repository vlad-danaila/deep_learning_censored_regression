import torch as t

LAYER_SIZE = 10

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