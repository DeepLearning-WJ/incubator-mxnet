from mxnet import nd
import mxnet as mx
from mxnet.gluon import nn
net = nn.Sequential()
# Add a sequence of layers.
net.add(# Similar to Dense, it is not necessary to specify the input channels
    # by the argument `in_channels`, which will be  automatically inferred
    # in the first forward pass. Also, we apply a relu activation on the
    # output. In addition, we can use a tuple to specify a  non-square
    # kernel size, such as `kernel_size=(2,4)`
    nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
    # One can also use a tuple to specify non-symmetric pool and stride sizes
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
    nn.MaxPool2D(pool_size=2, strides=2),
    # The dense layer will automatically reshape the 4-D output of last
    # max pooling layer into the 2-D shape: (x.shape[0], x.size/x.shape[0])
    nn.Dense(120, activation="relu"),
    nn.Dense(84, activation="relu"),
    nn.Dense(10))

print(net)
ctx = mx.gpu()
# ctx = mx.cpu()
net.initialize(ctx=ctx)
# Input shape is (batch_size, color_channels, height, width)
x = nd.random.uniform(shape=(4,1,28,28), ctx=ctx)
y = net(x)
print(y)
# print(y.shape)
print(net[0].weight.data().shape, net[5].bias.data().shape)
