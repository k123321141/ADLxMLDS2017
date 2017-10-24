from keras.models import *
from keras.layers import *

num_classes = 48
features_count = 39


def output(rnn_input,hidden_dim = 200,rnn_lay = SimpleRNN,bidirect = False,depth = 2,activation = 'tanh',dropout = 0.10):
    xx = rnn_input    
    if bidirect == True:
        for i in range(depth):
            xx = Bidirectional(rnn_lay(hidden_dim,activation=activation,return_sequences=True,consume_less ='mem'))(xx)
            xx = Dropout(dropout)(xx)

    else:
        for i in range(depth):
            xx = rnn_lay(hidden_dim,activation=activation,return_sequences=True,consume_less = 'mem')(xx)
            xx = Dropout(dropout)(xx)
    return xx

if __name__ == '__main__':
    import myinput
    import dic_processing
    from keras.utils import plot_model
    from keras.optimizers import *
    from keras.callbacks import *
#dic init setting,reshape
    max_len = 777
    dic1 = myinput.load_input('mfcc')
    dic_processing.pad_dic(dic1,max_len,max_len,num_classes)

    dic_processing.catogorate_dic(dic1,num_classes+1)

    x,y = dic_processing.toXY(dic1)
    num = x.shape[0]




    #model

    first_input = Input(shape=(max_len,features_count))
    rnn_input = BatchNormalization(input_shape = (max_len,features_count),axis = -1) (first_input)
    rnn_out = output(rnn_input,bidirect = True,depth = 2,hidden_dim = 200)

    result = TimeDistributed(Dense(num_classes+1,activation='softmax'))(rnn_out)

    model = Model(input = first_input,output = result)

    plot_model(model, to_file='../model.png',show_shapes = True)

    #construct sample matrix
    s_mat = np.zeros(y.shape[0:2],dtype = np.float32)
    np.place(s_mat,s_mat == 0,1)

    for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if y[i,j,-1] == 1:
                    s_mat[i,j:] = 0
                    break
    #
    sgd_opt = SGD(lr = 0.01)
    model.compile(loss='categorical_crossentropy', optimizer = sgd_opt,metrics=['accuracy'],sample_weight_mode = 'temporal')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    cks = ModelCheckpoint('../checkpoints/rnn.{epoch:02d}-{val_loss:.2f}.model',save_best_only=True,period = 5)
    model.fit(x,y,batch_size =100,epochs = 2000,callbacks=[early_stopping,cks],validation_split = 0.05,sample_weight = s_mat)
    
    print 'Done'
    model.save('../models/rnn.model')



