import mxnet as mx
from mxnet.gluon import Block, nn
from mxnet import ndarray as F


class Model(Block):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        # use name_scope to give child Blocks appropriate names.
        with self.name_scope():
            self.dense0 = nn.Dense(20)
            self.dense1 = nn.Dense(20)

    def forward(self, x):
        x = F.relu(self.dense0(x))
        return F.relu(self.dense1(x))


model = Model()
model.initialize(ctx=mx.cpu(0))
result = model(F.zeros((10, 10), ctx=mx.cpu(0)))
print(result)