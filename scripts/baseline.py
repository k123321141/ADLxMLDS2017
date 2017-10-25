from keras.models import *
from keras.layers import *
import rnn
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

    dic_processing.catogorate_dic(dic1,num_classes+1)
    x,y = dic_processing.toXY(dic1)
    num = x.shape[0]
    print x.shape,y.shape
    first_input = Input(shape=(max_len,features_count))
    rnn_input = first_input
    rnn_input = Masking()(rnn_input)
    #
    rnn_out = rnn.output(rnn_input,hidden_dim = 256,activation = 'tanh',bidirect = True,depth = 2)
    result = TimeDistributed(Dense(num_classes+1,activation='softmax'))(rnn_out)

    model = Model(input = first_input,output = result)
    plot_model(model, to_file='../model.png',show_shapes = True)

    #
    s_mat = np.ones(y.shape[0:2],dtype = np.float32)
    for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if y[i,j,-1] == 1:
                    s_mat[i,j+1:] = 0
                    s_mat[i,j] = 3
                    break
    #
    model.compile(loss='categorical_crossentropy',optimizer = Adam(0.001),metrics=['accuracy'],sample_weight_mode = 'temporal')
    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    cks = ModelCheckpoint('../checkpoints/baseline.{epoch:02d}-{val_loss:.2f}.model',save_best_only=True,period = 2)

    #model.fit(x,y,batch_size = 100,epochs = 2000,callbacks=[early_stopping,cks],validation_split = 0.05,sample_weight = s_mat)
    model.fit(x,y,batch_size = 100,epochs = 2000,callbacks=[early_stopping,cks],validation_split = 0.05)
    print 'Done'
