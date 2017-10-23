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

if __name__ == '__main__':
    import myinput
    import dic_processing
    from keras.utils import plot_model
    from keras.callbacks import *
#dic init setting,reshape
    dic1 = myinput.load_input('mfcc')
    dic_processing.pad_dic(dic1,max_len,max_len,num_classes)

    dic_processing.catogorate_dic(dic1,num_classes+1)

    x,y = dic_processing.toXY(dic1)
    num = x.shape[0]




#model

    x = x.reshape(num,max_len,features_count,1)

    cnn_input = Input(shape=(max_len,features_count,1))
    cnn_output = cnn_output(cnn_input)


    model = Model(input = cnn_input,output = cnn_output)

    plot_model(model, to_file='../model.png',show_shapes = True)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model.fit(x,y,batch_size = 10,epochs = 200,callbacks=[early_stopping],validation_split = 0.05)
    print 'Done'
    model.save('../models/cnn.model')
