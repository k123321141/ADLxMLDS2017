from keras.models import *
from keras.layers import *
import cnn,rnn


def seq_output(seq_input,max_out_len,hidden_dim=200,depth=1,activation='relu',rnn_lay = SimpleRNN,dropout = 0.15,bidirect = False):
#def output(rnn_input,hidden_dim = 200,rnn_lay = SimpleRNN,bidirect = False,depth = 2,activation = 'tanh',dropout = 0.10):
    
    #encoder
    if depth - 1 >= 1:
        #multi RNN layer encoding
        encoder_input = rnn.output(seq_input,rnn_lay = rnn_lay,bidirect = bidirect,activation = activation,depth = depth-1,hidden_dim=hidden_dim,dropout=dropout)
    else:
        encoder_input = seq_input
    if bidirect:
        encoder_out,hi_st = Bidirctional(rnn_lay(hidden_dim,return_state = True,return_sequences = False,activation =activation ))(encoder_input)
    else:
        encoder_out,hi_st = rnn_lay(hidden_dim,return_state = True,return_sequences = False,activation = activation)(encoder_input)

    #repeat
    decoder_input = RepeatVector(max_out_len)(encoder_out)
    #decoder
    decoder_output = rnn_lay(hidden_dim,return_sequences = True,activation=activation)(decoder_input,hi_st)
    if depth-1 >= 1:
        decoder_output = rnn.output(decoder_output,rnn_lay = rnn_lay,bidirect = bidirect,activation = activation,depth=depth,hidden_dim=hidden_dim,dropout=dropout)

    return decoder_output

if __name__ == '__main__':
    import myinput
    import dic_processing
    from keras.utils import plot_model
    from keras.callbacks import *
    from keras.optimizers import *
#dic init setting,reshape
    max_len = 777
    num_classes = 48
    features_count = 39
    max_out_len = 80
    seq = True
    dic1 = myinput.load_input('mfcc')
    if seq == True:
        seq_dict = myinput.read_seq_Y('../data//mfcc/seq_y.lab')
        for sentenceID in sorted(seq_dict.keys()):
            frame_dic = seq_dict[sentenceID]
            seq_y = myinput.dic2ndarray(frame_dic)
            seq_y = seq_y.reshape(seq_y.shape[0],1)

            x,y = dic1[sentenceID]
            dic1[sentenceID] = x,seq_y
        dic_processing.pad_dic(dic1,max_len,max_out_len,num_classes)
    else:
        dic_processing.pad_dic(dic1,max_len,max_len,num_classes)

    x,y = dic_processing.toXY(dic1)
    num = x.shape[0]

    x = x.reshape(num,max_len,features_count,1)
    #(777,39) -> (777,39,1)


    first_input = Input(shape=(max_len,features_count,1))
    cnn_input = BatchNormalization(axis = -2) (first_input)         #the axis of nomaliztion is -2 (3696,777,39,1)
    cnn_output = cnn.output(cnn_input,kernel_size =(3,5),depth = 1,filters = 10,padding ='valid')
    #(777,39,1) -> (775,35,10)
    seq_input = Reshape((775,35*10))(cnn_output)
    #
    seq_output = seq_output(seq_input,max_out_len,hidden_dim=200,depth=1,activation='relu',bidirect = False) 
    #
    result = TimeDistributed(Dense(num_classes+1,activation='softmax'))(seq_output)

    model = Model(input = first_input,output = result)

    plot_model(model, to_file='../model.png',show_shapes = True)

    #
    s_mat = np.zeros(y.shape[0:2],dtype = np.float32)
    np.place(s_mat,s_mat == 0,1)
    for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if y[i,j,-1] == 1:
                    s_mat[i,j:] = 0
                    break
    #
    sgd_opt = SGD(lr = 0.01)
    print x.shape,y.shape
    model.compile(loss='categorical_crossentropy', optimizer=sgd_opt,metrics=['accuracy'],sample_weight_mode = 'temporal')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    cks = ModelCheckpoint('../checkpoints/seq.{epoch:02d}-{val_loss:.2f}.model',save_best_only=True,period = 2)
    model.fit(x,y,batch_size =100,epochs = 2000,callbacks=[early_stopping,cks],validation_split = 0.05,sample_weight = s_mat)
    print 'Done'
    model.save('../models/seq.model')
