import mxnet as mx
from mxnet import profiler
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from mxnet import autograd
class MyAddOne(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0]+1)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_grad[0])

@mx.operator.register('MyAddOne')
class CustomAddOneProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(CustomAddOneProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return [in_shape[0]], [in_shape[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return MyAddOne()


inp = mx.nd.zeros(shape=(500, 500))

# profiler.set_config(profile_all=True, continuous_dump=True, \
#                     aggregate_stats=True)
# profiler.set_state('run')
#
# w = nd.Custom(inp, op_type="MyAddOne")
#
# mx.nd.waitall()
#
# profiler.set_state('stop')
# print(profiler.dumps())
# profiler.dump(finished=False)

# Set profile_all to True
profiler.set_config(profile_all=True, aggregate_stats=True, continuous_dump=True)
# OR, Explicitly Set profile_symbolic and profile_imperative to True
profiler.set_config(profile_symbolic=True, profile_imperative=True, \
                    aggregate_stats=True, continuous_dump=True)

profiler.set_state('run')
# Use Symbolic Mode
a = mx.symbol.Variable('a')
b = mx.symbol.Custom(data=a, op_type='MyAddOne')
c = b.bind(mx.cpu(), {'a': inp})
y = c.forward()
mx.nd.waitall()
profiler.set_state('stop')
print(profiler.dumps())
profiler.dump()
