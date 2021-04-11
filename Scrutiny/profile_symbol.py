import mxnet as mx
from mxnet import profiler
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from mxnet import autograd
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