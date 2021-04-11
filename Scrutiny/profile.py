import mxnet as mx
from mxnet import profiler
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from mxnet import autograd

profiler.set_config(profile_all=False,
                    profile_symbolic=True,
                    profile_imperative=True,
                    profile_memory=False,
                    profile_api=False,
                    aggregate_stats=True,
                    continuous_dump=True,
                    filename='profile_output.json')

net = gluon.nn.HybridSequential()
with net.name_scope():
    net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(512, activation="relu"))
    net.add(gluon.nn.Dense(10))


dataset = gluon.data.vision.MNIST(train=True)
dataset = dataset.transform_first(transforms.ToTensor())
dataloader = gluon.data.DataLoader(dataset, batch_size=64, shuffle=True)

# Use GPU if available
if mx.context.num_gpus():
    ctx=mx.gpu()
else:
    ctx=mx.cpu()

# Initialize the parameters with random weights
net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

# Use SGD optimizer
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

# Softmax Cross Entropy is a frequently used loss function for multi-class classification
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# A helper function to run one training iteration
def run_training_iteration(data, label):
    # Load data and label is the right context
    data = data.as_in_context(ctx)
    label = label.as_in_context(ctx)
    # Run the forward pass
    with autograd.record():
        output = net(data)
        loss = softmax_cross_entropy(output, label)
    # Run the backward pass
    # loss.backward()
    # Apply changes to parameters
    # trainer.step(data.shape[0])

# Run the first iteration without profiling
itr = iter(dataloader)
# run_training_iteration(*next(itr))

# data, label = next(itr)

# Ask the profiler to start recording
profiler.set_state('run')

run_training_iteration(*next(itr))

# Make sure all operations have completed
mx.nd.waitall()
# Ask the profiler to stop recording
profiler.set_state('stop')
# Dump all results to log file before download
profiler.dump(finished=False)# 浏览器查看
# 在终端打印
print(profiler.dumps())
