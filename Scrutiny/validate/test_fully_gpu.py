'''
https://mxnet.incubator.apache.org/versions/1.7.0/api/python/docs/tutorials/getting-started/crash-course/2-nn.html
'''
from mxnet import nd
import mxnet as mx
from mxnet.gluon import nn

if len(mx.test_utils.list_gpus())!=0:
    ctx = mx.gpu()
else:
    ctx = mx.cpu

layer = nn.Dense(2)
layer.initialize(ctx=ctx)
print(layer.params)
x = nd.random.uniform(-1,1,(3,4), ctx=ctx)
print(x)
# print(x.shape)
y = layer(x)
# print(y.shape)
print(y)
print(layer.weight.data())
print(layer.bias.data())
