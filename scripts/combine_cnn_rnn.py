from keras.models import *
from keras.layers import *

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
    
    dic_processing.pad_dic(dic1,max_len,max_len,num_classes)

    dic_processing.catogorate_dic(dic1,num_classes+1)
    x,y = dic_processing.toXY(dic1)
    num = x.shape[0]
    y = y[:,1:-1,:]
    x = x.reshape(num,max_len,features_count,1)

    first_input = Input(shape=(max_len,features_count,1))
    cnn_input = first_input
    cnn_output = Conv2D(10,kernel_size = (3,5),use_bias = False,activation = 'tanh')(cnn_input)
    #(777,39,1) -> (775,35,10)
    seq_input = Reshape((775,35*10))(cnn_output)
    #
    seq_input = Masking()(seq_input)
    #
    rnn_out = Bidirectional(GRU(200,activation = 'tanh',return_sequences = True))(seq_input)
    result = TimeDistributed(Dense(num_classes+1,activation='softmax'))(rnn_out)

    model = Model(input = first_input,output = result)
    plot_model(model, to_file='../model.png',show_shapes = True)
#    model.load_weights('../checkpoints/seq2.00-3.64.model')

    #
    s_mat = np.ones(y.shape[0:2],dtype = np.float32)
    for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if y[i,j,-1] == 1:
                    s_mat[i,j+1:] = 0
                    s_mat[i,j] = 3
                    break
    #
    sgd_opt = SGD(lr = 0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd_opt,metrics=['accuracy'],sample_weight_mode = 'temporal')
    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    cks = ModelCheckpoint('../checkpoints/cnn+rnn_alignment.{epoch:02d}-{val_loss:.2f}.model',save_best_only=True,period = 1)

    model.fit(x,y,batch_size = 100,epochs = 2000,callbacks=[early_stopping,cks],validation_split = 0.05,sample_weight = s_mat)
    print 'Done'
