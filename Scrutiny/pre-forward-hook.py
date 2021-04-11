import mxnet as mx
from mxnet.gluon import nn

block = nn.Dense(10)
block.initialize()
print("{}".format(block))
# Dense(None -> 10, linear)

def pre_hook(block, input) -> None:  # notice it has two arguments, one block and one input
    print("{}".format(block))
    return

# register
pre_handle = block.register_forward_pre_hook(pre_hook)
input = mx.nd.ones((3, 5))
print(block(input))

# Dense(None -> 10, linear)
# [[ 0.11254273  0.11162187  0.02200389 -0.04842059  0.09531345  0.00880495
#  -0.07610667  0.1562067   0.14192852  0.04463106]
# [ 0.11254273  0.11162187  0.02200389 -0.04842059  0.09531345  0.00880495
#  -0.07610667  0.1562067   0.14192852  0.04463106]
# [ 0.11254273  0.11162187  0.02200389 -0.04842059  0.09531345  0.00880495
#  -0.07610667  0.1562067   0.14192852  0.04463106]]
# <NDArray 3x10 @cpu(0)>
