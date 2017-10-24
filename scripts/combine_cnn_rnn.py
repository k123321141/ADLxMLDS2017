from keras.models import *
from keras.layers import *
import myinput
import dic_processing
from keras.utils import plot_model
from keras.callbacks import *
from keras.optimizers import *
import rnn
import cnn
num_classes = 48
features_count = 39

if __name__ == '__main__':
    max_len = 777
    dic1 = myinput.load_input('mfcc')
    dic_processing.pad_dic(dic1,max_len,max_len,num_classes)

    dic_processing.catogorate_dic(dic1,num_classes+1)

    x,y = dic_processing.toXY(dic1)
    num = x.shape[0]




#model

    x = x.reshape(num,max_len,features_count,1)
    #(777,39) -> (777,39,1)
    y = y[:,1:-1,:]
    #(3696,777,49) -> (3696,775,49)
    print y.shape 


    first_input = Input(shape=(max_len,features_count,1))
    cnn_input = BatchNormalization(axis = -2) (first_input)         #the axis of nomaliztion is -2 (3696,777,39,1)
    cnn_output = cnn.cnn_output(cnn_input,kernel_size =(3,5),depth = 1,filters = 10,padding ='valid')
    #(777,39,1) -> (775,35,10)
    
    rnn_input = Reshape((775,35*10))(cnn_output)
    rnn.
    rnn_output = rnn.rnnoutput(rnn_input,)
    result = TimeDistributed(Dense(num_classes+1,activation='softmax'))(result)

    model = Model(input = first_input,output = result)

    plot_model(model, to_file='../model.png',show_shapes = True)
    sgd_opt = SGD(lr = 0.01)

    model.compile(loss='categorical_crossentropy', optimizer=sgd_opt,metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model.fit(x,y,batch_size = 10,epochs = 200,callbacks=[early_stopping],validation_split = 0.05)
    print 'Done'
    model.save('../models/cnn.model')
