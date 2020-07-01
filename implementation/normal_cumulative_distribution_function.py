from implementation.util import to_numpy, to_torch
import torch as t
from scipy.stats import norm
import numpy as np
from implementation.util import normalize

class __CDF(t.autograd.Function):

    @staticmethod
    def forward(ctx, x: t.Tensor) -> t.Tensor:
        _x = to_numpy(x)
        pdf = to_torch(norm.pdf(_x), grad = False)
        ctx.save_for_backward(pdf)
        return to_torch(norm.cdf(_x), grad = False)

    @staticmethod
    def backward(ctx, grad_output):
        pdf = ctx.saved_tensors[0]
        grad = None
        if ctx.needs_input_grad[0]:
            grad = grad_output * pdf
        return grad

cdf = __CDF.apply

if __name__ == '__main__':
    input = [10, 15, 20, 25, 30]

    # manual gradient computing
    x = np.array(input)
    mean, std = x.mean(), x.std()
    x_normalized = normalize(x, mean, std)
    expected_cdf = norm.cdf(x_normalized)
    expected_log_likelihood = np.log(expected_cdf)
    expected_grad_log_likelihood_by_x = norm.pdf(x_normalized) / (expected_cdf * std)

    # automatic gradient computing
    x = to_torch(input, grad=True)
    # in this test mean & std are considered constants
    x_normalized = normalize(x, mean, std)
    cdf_result = cdf(x_normalized)
    # assert_almost_equal(to_numpy(cdf_result), expected_cdf)

    log_likelihood_result = t.log(cdf_result)
    # assert

    loss = t.sum(log_likelihood_result)
    loss.backward()
    print(x.grad, expected_grad_log_likelihood_by_x)
    # assert_almost_equal(to_numpy(x.grad), expected_grad_cdf_by_x)