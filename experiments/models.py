import torch as t
from torch.nn import Linear, BatchNorm1d, ReLU

def get_dense_net(nb_layers, input_size, hidden_size):
    if nb_layers == 1:
        return t.nn.Linear(input_size, 1)
    sequential = t.nn.Sequential(Linear(input_size, hidden_size), BatchNorm1d(hidden_size, affine = False), ReLU())
    # start from 1 since the input layer is already setup
    # end at -1 since we setup the output layer separately
    for i in range(1, nb_layers - 1):
        sequential.append(Linear(hidden_size, hidden_size))
        sequential.append(BatchNorm1d(hidden_size, affine = False))
        sequential.append(ReLU())
    sequential.append(Linear(hidden_size, 1))
    return sequential

# Testing
if __name__ == '__main__':
    x = t.tensor([[1., 2., 3.], [4., 5., 6.]])

    # m = get_dense_net(1, 3, 10)
    # m = get_dense_net(2, 3, 10)
    # m = get_dense_net(3, 3, 10)
    m = get_dense_net(4, 3, 10)

    print(m(x))