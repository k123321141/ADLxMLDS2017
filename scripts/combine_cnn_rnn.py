from keras.models import *
from keras.layers import *
import cnn,rnn

num_classes = 48
features_count = 39


if __name__ == '__main__':
    import myinput
    import dic_processing
    from keras.utils import plot_model
    from keras.callbacks import *
    from keras.optimizers import *
#dic init setting,reshape
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
    cnn_output = cnn.output(cnn_input,kernel_size =(3,5),depth = 1,filters = 10,padding ='valid')
    #(777,39,1) -> (775,35,10)
    rnn_input = Reshape((775,35*10))(cnn_output)
    #
    rnn_output = rnn.output(rnn_input,bidirect = True,depth = 2,)
    #
    result = TimeDistributed(Dense(num_classes+1,activation='softmax'))(rnn_output)

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

    model.compile(loss='categorical_crossentropy', optimizer=sgd_opt,metrics=['accuracy'],sample_weight_mode = 'temporal')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    cks = ModelCheckpoint('../checkpoints/combine.{epoch:02d}-{val_loss:.2f}.model',save_best_only=True,period = 5)
    model.fit(x,y,batch_size =100,epochs = 2000,callbacks=[early_stopping,cks],validation_split = 0.05,sample_weight = s_mat)
    print 'Done'
    model.save('../models/cnn+rnn.model')
