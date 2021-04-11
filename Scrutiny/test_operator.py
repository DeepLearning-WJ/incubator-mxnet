import mxnet as mx
import numpy as np
from mxnet.test_utils import check_symbolic_forward, check_symbolic_backward, check_numeric_gradient, rand_shape_nd, assert_almost_equal
# a = mx.sym.Variable('a', shape=(2, 0))
# b = mx.sym.Variable('b')
# c = mx.sym.Variable('c', shape=(0, 3))
# d = a * b + b * c
# print(d.infer_shape())
# print(a.shape())

def test_quadratic_function():
    def f(x, a, b, c):
        return a * x**2 + b * x + c

    a = np.random.random_sample()
    b = np.random.random_sample()
    c = np.random.random_sample()
    data = mx.symbol.Variable('data')
    quad_sym = mx.sym.contrib.quadratic(data=data, a=a, b=b, c=c)
    print(quad_sym)
    for dtype in [np.float16, np.float32, np.float64]:
        for ndim in range(1, 6):
            shape = rand_shape_nd(ndim, 5)
            data_np = np.random.randn(*shape).astype(dtype)
            expected = f(data_np, a, b, c)
            backward_expected = 2 * a * data_np + b

            # check imperative forward
            output = mx.nd.contrib.quadratic(mx.nd.array(data_np), a=a, b=b, c=c)
            assert_almost_equal(output.asnumpy(),expected,
                                rtol=1e-2 if dtype is np.float16 else 1e-5,
                                atol=1e-2 if dtype is np.float16 else 1e-5)
            # check forward
            check_symbolic_forward(quad_sym, [data_np], [expected],
                                    rtol=1e-2 if dtype is np.float16 else 1e-5,
                                    atol=1e-2 if dtype is np.float16 else 1e-5)
            # check backward
            check_symbolic_backward(quad_sym, [data_np], [np.ones(expected.shape)],
                                        [backward_expected],
                                        rtol=1e-2 if dtype is np.float16 else 1e-5,
                                        atol=1e-2 if dtype is np.float16 else 1e-5)
            # check backward using finite difference
            check_numeric_gradient(quad_sym, [data_np], atol=0.001)
test_quadratic_function()
