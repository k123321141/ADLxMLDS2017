from keras.models import *
from keras.layers import *

def seq_model(hidden_dim,x,max_out_len,num_classes,depth=1,activation='tanh',rnn_lay = SimpleRNN,dropout = 0.25):

    #encoder
    if rnn_lay == LSTM:
        x,hidden_state,un = rnn_lay(hidden_dim,activation = activation,return_state = True)(x)[0:-1]
    else:
        x,hidden_state = rnn_lay(hidden_dim,activation = activation,return_state = True)(x)
    #
    x = RepeatVector(max_out_len)(x)
    #decoder
    #first decoder layer
    x = rnn_lay(hidden_dim,return_sequences = True,activation = activation)(x,initial_state = hidden_state)
    x = Dropout(dropout)(x)
    for i in range(depth-1):
        x = rnn_lay(hidden_dim,return_sequences = True,activation = activation)(x)
        x = Dropout(dropout)(x)
    x = TimeDistributed(Dense(num_classes,activation = 'softmax'))(x)
    return x



