from keras.models import *
from keras.layers import *
from keras.utils import to_categorical
import keras.backend as K
def ctc_lambda_func(y_true, y_pred):
    return K.ctc_batch_cost(y_true, y_pred, 777, 80)

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
    buf_x = []
    buf_y = []
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
       for sentenceID in sorted(dic1.keys()):
            x,y = dic1[sentenceID]
            buf_x.append(x)
            buf_y.append(y)
    dic_processing.catogorate_dic(dic1,num_classes+1)
    x,y = dic_processing.toXY(dic1)
    num = x.shape[0]
    
    print x.shape
    x = x.reshape(num,max_len,features_count,1)
    first_input = Input(shape=(max_len,features_count,1))
    cnn_input = BatchNormalization(axis = -1)(first_input)
    cnn_output = Conv2D(30,kernel_size = (3,5),use_bias = False,activation = 'relu',padding = 'valid')(cnn_input)
    #(87,39,1) -> (85,35,10)
    seq_input = Reshape((max_len-2,35*30))(cnn_output)
    seq_input = Masking()(seq_input)
    #
    rnn_lay = LSTM

    #xx,state_h, state_c = (rnn_lay(300,activation = 'tanh',return_state = True))(seq_input)
    #bidirection
    x1,state_h, state_c  = rnn_lay(300,activation = 'tanh',return_state = True)(seq_input)
    x2,state_h, state_c  = rnn_lay(300,activation = 'tanh',return_state = True,go_backwards = True)(seq_input,[state_h, state_c])
    xx = Concatenate(axis = -1)([x1,x2])

    xx = RepeatVector(max_out_len)(xx)
    x1,state_h, state_c  = rnn_lay(300,activation = 'tanh',return_state = True,return_sequences = True)(xx,[state_h, state_c])
    x2 = rnn_lay(300,activation = 'tanh',return_state = False,return_sequences = True,go_backwards = True)(xx,[state_h, state_c])
    xx = Concatenate(axis = -1)([x1,x2])
    
    xx = Dropout(0.25)(xx)
    result = TimeDistributed(Dense(num_classes+1,activation='softmax'))(xx)

    model = Model(input = first_input,output = result)
    plot_model(model, to_file='../model.png',show_shapes = True)
    #model.load_weights('../checkpoints/simple.18-1.65.model')

    s_mat = np.ones(y.shape[0:2],dtype = np.float32)
    for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if y[i,j,-1] == 1:
                    end = np.min([j+4,y.shape[1]])
                    s_mat[i,j+1:end] = 1
                    s_mat[i,j] = 5
                    s_mat[i,end:] = 0
                    break
                elif y[i,j,37] == 1: #sil
                    s_mat[i,j] = 0.5

    opt = Adam(lr = 0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'],sample_weight_mode = 'temporal')
    #model.compile(loss=ctc_lambda_func, optimizer=sgd_opt,metrics=['accuracy'],sample_weight_mode = 'temporal')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    cks = ModelCheckpoint('../checkpoints/seq.{epoch:02d}-{val_loss:.2f}.cks',save_best_only=True,period = 2)

    model.fit(x,y,batch_size = 30,epochs = 2000,callbacks=[early_stopping,cks],validation_split = 0.05,sample_weight = s_mat)
    #model.fit(x,y,batch_size = 30,epochs = 2000,callbacks=[early_stopping,cks],validation_split = 0.05)
    print 'Done'
