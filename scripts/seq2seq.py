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
    seq = False
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

    x = np.vstack(buf_x)
    y = np.vstack(buf_y)
    y = to_categorical(y,num_classes)
    print x.shape
    num = x.shape[0]
    bat_len = 87
    
    x = x.reshape(num/bat_len,bat_len,39,1)
    y = y.reshape(num/bat_len,bat_len,num_classes)
    '''
    x = x.reshape(1,num,features_count,1)
    y = y.reshape(1,num,num_classes)
    '''
    print x.shape,y.shape
    
    y = y[:,1:-1,:]
    #(777,39) -> (777,39,1)
    print x.shape

    first_input = Input(shape=(bat_len,features_count,1))
    cnn_input = first_input
    cnn_output = Conv2D(10,kernel_size = (3,5),use_bias = False,activation = 'tanh')(cnn_input)
    #(87,39,1) -> (85,35,10)
    seq_input = Reshape((85,35*10))(cnn_output)
    #
    '''
    xx,st = SimpleRNN(200,activation = 'tanh',return_state = True)(seq_input)
    xx = RepeatVector(max_out_len)(xx)
    xx = SimpleRNN(200,activation = 'tanh',return_sequences = True)(xx,initial_state = st)
    '''
    xx = SimpleRNN(200,activation = 'tanh',return_sequences = True)(seq_input)
    result = TimeDistributed(Dense(num_classes,activation='softmax'))(xx)

    model = Model(input = first_input,output = result)
    plot_model(model, to_file='../model.png',show_shapes = True)
    #model.load_weights('../checkpoints/simple.18-1.65.model')

    '''
    s_mat = np.ones(y.shape[0:2],dtype = np.float32)
    for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if y[i,j,-1] == 1:
                    s_mat[i,j+1:] = 0.2
                    s_mat[i,j] = 3
                    break
    '''
    sgd_opt = SGD(lr = 0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd_opt,metrics=['accuracy'])
    #model.compile(loss=ctc_lambda_func, optimizer=sgd_opt,metrics=['accuracy'],sample_weight_mode = 'temporal')
    early_stopping = EarlyStopping(monitor='val_loss', patience=100)
    cks = ModelCheckpoint('../checkpoints/test.{epoch:02d}-{val_loss:.2f}.model',save_best_only=True,period = 10)

    #model.fit(x,y,batch_size = 300,epochs = 2000,callbacks=[early_stopping,cks],validation_split = 0.05,sample_weight = s_mat)
    model.fit(x,y,batch_size = 300,epochs = 2000,callbacks=[early_stopping,cks],validation_split = 0.05)
    print 'Done'
