from keras.models import *
from keras.layers import *

def seq_output(seq_input,max_out_len,num_classes,hidden_dim=200,depth=1,activation='tanh',rnn_lay = SimpleRNN,dropout = 0.25):

    #encoder
    x,hidden_st = rnn_lay(hidden_dim,return_sequences = False,return_state = True,go_backwards = True)(seq_input)
    x = RepeatVector(max_out_len)(x)
    #decoder
    x = rnn_lay(hidden_dim,return_sequences = True,activation = activation)(x,initial_state = hidden_st)
    x = Dropout(dropout)(x)
    x = TimeDistributed(Dense(num_classes,activation = 'softmax'))(x)
    return x



