from keras.models import *
from keras.layers import *

num_classes = 48
features_count = 39

max_len = 777

bi = True
rnn_lay = SimpleRNN
def cnn_output(xx):


    xx = Conv2D(30,input_shape = (max_len,features_count),kernel_size = (1,5),padding='valid',activation = 'relu',data_format = 'channels_last')(xx) 
    #777,35,30
    xx = Conv2D(60,kernel_size = (1,5),padding='valid',activation = 'relu',data_format = 'channels_last')(xx) 
    #777,31,60
    xx = BatchNormalization(axis=-1 )(xx)

    xx = Reshape((max_len,31*60))(xx)
    xx = Masking(mask_value=0)(xx)
    if bi == True:

        xx = Bidirectional(rnn_lay(60,activation='tanh',return_sequences=True,implementation=1))(xx)
        xx = Bidirectional(rnn_lay(num_classes+1,activation='tanh',return_sequences=True,implementation=1))(xx)
        xx = TimeDistributed(Dense(num_classes+1,activation = 'softmax'))(xx)
    else:
        xx = Bidirectional(rnn_lay(60,activation='tanh',return_sequences=True,implementation=1))(xx)
        xx = rnn_lay(40,activation='tanh',return_sequences=True,implementation=1)(xx)
        xx = rnn_lay(num_classes+1,activation='softmax',return_sequences=True,implementation=1)(xx)

    return xx
