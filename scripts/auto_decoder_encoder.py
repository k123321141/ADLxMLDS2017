from keras.models import *
from keras.layers import *

def seq_output(seq_input,max_out_len,num_classes,hidden_dim=200,depth=1,activation='relu',rnn_lay = SimpleRNN,dropout = 0.25):

    #encoder
    x = Masking(mask_value=48)
    x = Reshape((777,49,1))(seq_input)
    x = Conv2D(10,kernel_size = (1,6),activation = activation)(x)
    x = Conv2D(20,kernel_size = (1,6),activation = activation)(x)
    x = Reshape((777,39*20))(x)
    x = rnn_lay(hidden_dim,activation = activation,return_sequences = True,return_state = False,go_backwards = True)(x)
    x,hidden_st = rnn_lay(hidden_dim,return_sequences = False,return_state = True)(x)

    #x,hidden_st = rnn_lay(hidden_dim,return_sequences = False,return_state = True,go_backwards = True)(seq_input)
    x = RepeatVector(max_out_len)(x)
    #decoder
    x = rnn_lay(hidden_dim,return_sequences = True,activation = activation)(x,initial_state = hidden_st)
    x = rnn_lay(hidden_dim,return_sequences = True,activation = activation)(x)
        
    #x = Dropout(dropout)(x)
    x = TimeDistributed(Dense(num_classes,activation = 'softmax'))(x)
    return x



