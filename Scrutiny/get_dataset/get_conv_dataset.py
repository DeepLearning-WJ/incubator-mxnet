from mxnet import nd
import mxnet as mx
from mxnet import profiler
from mxnet.gluon import nn
import pandas as pd

df=pd.DataFrame(columns={'batch', 'input_channel', 'output_channel', 'height', \
                            'kernel', 'stride','time'})


for b in range(1,10):
    for input in range(1,10):
        for out in range(1,10):
            for k in range(1,10):
                for height in range(k,10):
                    for s in range(1,10):
                        profiler.set_config(profile_all=False,
                                            profile_symbolic=False,
                                            profile_imperative=True,
                                            profile_memory=False,
                                            profile_api=False,
                                            aggregate_stats=True,
                                            continuous_dump=True,
                                            filename='profile_output.json')
                        profiler.set_state('run')
                        net = nn.Sequential()
                        result = ''

                        net.add(nn.Conv2D(channels=out, kernel_size=k, strides=s))

                        print(net)
                        ctx = mx.gpu()
                        net.initialize(ctx=ctx)
                        for i in range(10000):
                            x = nd.random.uniform(shape=(b,input,height,height), ctx=ctx)
                            y = net(x)
                            # print(y.shape)
                        mx.nd.waitall()
                        profiler.set_state('stop')
                        profiler.dump(finished=False)
                        f = open("./dumps.txt", 'w+')
                        print(profiler.dumps())

                        # f = open("./dumps.txt", 'r')
                        # data = f.readline()
                        # print(data)
                        #
                        # while data:
                        #     list = data.split()
                        #     if len(list) == 0:
                        #         data = f.readline()
                        #         continue
                        #     if list[0] == 'Convolution':
                        #         print(list[-1])
                        #         result=list[-1]
                        #     data = f.readline()
                        #
                        # f.close()
                        # config = {'batch': [b], 'input_channel': [input], \
                        #         'output_channel': [out], 'height': [height], \
                        #         'kernel': [k], 'stride': [s],'time':[result]}
                        # df1=pd.DataFrame(config)
                        # df1.to_csv('result.csv',header=False, index=False,mode='a')
