from keras.models import *
from keras.layers import *
num_classes = 48

max_out_len = 80

hidden_dim = 200
max_out_len = 80


def seq_output(x):

    #seq_input = Input(shape=(max_len,num_classes+1))
    


    #decoder
    rnn_lay = SimpleRNN
    x,hidden_st = rnn_lay(hidden_dim,return_sequences = False,return_state = True,go_backwards = True)(seq_input)
    #repeat

    x = RepeatVector(max_out_len)(x)
    #encoder
    x = rnn_lay(hidden_dim,return_sequences = True,return_state = False,go_backwards = False)(x)
    x = TimeDistributed(Dense(num_classes+1),activation = 'softmax')(x)
    
    return x

