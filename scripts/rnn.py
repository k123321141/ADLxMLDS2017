from keras.models import *
from keras.layers import *

num_classes = 48
features_count = 39


def rnn_output(rnn_input,hidden_dim = 200,rnn_lay = SimpleRNN,bidirect = False,depth = 2,activation = 'tanh'):
    xx = rnn_input    
    if bidirect == True:
        for i in range(depth):
            xx = Bidirectional(rnn_lay(hidden_dim,activation=activation,return_sequences=True,consume_less ='mem'))(xx)

    else:
        for i in range(depth):
            xx = rnn_lay(hidden_dim,activation=activation,return_sequences=True,consume_less = 'mem')(xx)
    return xx

if __name__ == '__main__':
    import myinput
    import dic_processing
    from keras.utils import plot_model
    from keras.callbacks import *
#dic init setting,reshape
    max_len = 777
    dic1 = myinput.load_input('mfcc')
    dic_processing.pad_dic(dic1,max_len,max_len,num_classes)

    dic_processing.catogorate_dic(dic1,num_classes+1)

    x,y = dic_processing.toXY(dic1)
    num = x.shape[0]




#model

    x = x.reshape(num,max_len,features_count)

    rnn_input = Input(shape=(max_len,features_count))
    rnn_out = rnn_output(rnn_input)


    model = Model(input = rnn_input,output = rnn_out)

    plot_model(model, to_file='../model.png',show_shapes = True)

    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model.fit(x,y,batch_size = 400,epochs = 200,callbacks=[early_stopping],validation_split = 0.05)
    print 'Done'
    model.save('../models/rnn.model')






