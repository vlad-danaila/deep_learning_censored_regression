import torch as t
from torch.nn import Linear, BatchNorm1d, ReLU, Dropout
from experiments.constants import IS_CUDA_AVILABLE

def get_dense_net(nb_layers, input_size, hidden_size, dropout_rate):
    if nb_layers == 1:
        liniar_model = t.nn.DataParallel(t.nn.Linear(input_size, 1))
        if IS_CUDA_AVILABLE:
            liniar_model = liniar_model.cuda()
        return liniar_model
    sequential = t.nn.Sequential(
        Linear(input_size, hidden_size), BatchNorm1d(hidden_size, affine = False), ReLU(), Dropout(p = dropout_rate)
    )
    # start from 1 since the input layer is already setup
    # end at -1 since we setup the output layer separately
    for i in range(1, nb_layers - 1):
        sequential.append(Linear(hidden_size, hidden_size))
        sequential.append(BatchNorm1d(hidden_size, affine = False))
        sequential.append(ReLU())
        sequential.append(Dropout(p = dropout_rate))
    sequential.append(Linear(hidden_size, 1))
    if IS_CUDA_AVILABLE:
        sequential = sequential.cuda()
    sequential = t.nn.DataParallel(sequential)
    return sequential

# Testing
if __name__ == '__main__':
    x = t.tensor([[1., 2., 3.], [4., 5., 6.]])

    # m = get_dense_net(1, 3, None, None)
    # m = get_dense_net(2, 3, 10, .2)
    # m = get_dense_net(3, 3, 10, .2)
    m = get_dense_net(4, 3, 10, .2)

    print(m(x))