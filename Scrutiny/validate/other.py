'''
https://mxnet.incubator.apache.org/versions/1.7.0/api/python/docs/tutorials/getting-started/crash-course/2-nn.html
'''
from mxnet import nd
import mxnet as mx
from mxnet.gluon import nn

class MixMLP(nn.Block):
    def __init__(self, **kwargs):
        # Run `nn.Block`'s init method
        super(MixMLP, self).__init__(**kwargs)
        self.blk = nn.Sequential()
        self.blk.add(nn.Dense(3, activation='relu'),
                     nn.Dense(4, activation='relu'))
        self.dense = nn.Dense(5)
    def forward(self, x):
        y = nd.relu(self.blk(x))
        print(y)
        return self.dense(y)

net = MixMLP()
print(net)

net.initialize()
x = nd.random.uniform(shape=(2,2))
print(net(x))
print(net.blk[1].weight.data())