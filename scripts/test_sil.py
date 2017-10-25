#from keras.models import *
#from keras.layers import *
#from keras.callbacks import *
#from keras.utils import plot_model
import myinput
import numpy as np
import dic_processing
#import cnn
#import auto_decoder_encoder

max_len = 777
num_classes = 48
features_count = 39
max_out_len = 80
seq = True

#dic init setting,reshape
dic1 = myinput.load_input('mfcc')
#dic_processing.pad_dic(dic1,max_len,max_len,num_classes)

buf = []
for k in dic1.keys():
    x,y = dic1[k]

    buf.append(x.shape[0])
#    np.place(y, y==37, 1)
#
#    np.place(y, y!=1, 0)
#    
#    sil_start = 0 if y[0] == 1 else -1
#    for j in range(1,max_len,1):
#        if y[j] == 1 and sil_start == -1:      #start of sil slice
#            sil_start = j
#        elif y[j] != 1 and sil_start != -1:    #end of sil slice
#            end = j
#            mid = (sil_start + end)/2
#            y[mid] = 2
#            sil_start = -1
#
#    np.place(y, y!=2, 0)
#    np.place(y, y!=0, 1)
#
#    dic1[k] = x,y
#
print sorted(buf)
print np.min(buf),np.max(buf),np.mean(buf),np.var(buf)

#dic_processing.catogorate_dic(dic1,0)
#x,y = dic_processing.toXY(dic1)
#
#num = x.shape[0]
#
#print 'start'
#
#model = Sequential()
#model.add(SimpleRNN(256,return_sequences = True,activation='sigmoid',input_dim = 39))
#model.add(Dropout(0.25))
#model.add(TimeDistributed(Dense(256,activation='sigmoid')))
#model.add(Dropout(0.25))
#model.add(TimeDistributed(Dense(1,activation='sigmoid')))
#
#s_mat = np.ones(y.shape[0:2],dtype = np.float32)
#for i in range(y.shape[0]):
#    for j in range(y.shape[1]):
#        if y[i,j] == 1:
#            s_mat[i,j] = 1000
#            break
##
#
#
#wd = {0:1,1:1000}
#
#model.compile(loss='binary_crossentropy', optimizer='sgd',metrics=['accuracy'],sample_weight_mode='temporal')
#early_stopping = EarlyStopping(monitor='val_loss', patience=2)
#model.fit(x,y,batch_size = 100,epochs = 200,callbacks=[early_stopping],validation_split = 0.05,sample_weight = s_mat)
#print 'Done'
#model.save('../models/sil_cpu.model')
