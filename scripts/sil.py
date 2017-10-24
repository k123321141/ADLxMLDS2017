import myinput
from keras.models import *
from keras.layers import *
from keras.utils import *
from keras.callbacks import *
from keras.optimizers import *
import numpy as np
import random
import dic_processing
num_classes = 48
max_len = 777

#dic init setting,reshape
dic1 = myinput.load_input('mfcc')
#change y with seq_y
for k in dic1.keys():
    x,y = dic1[k]
    
    np.place(y, y==37, 1)
    '''
    np.place(y, y==9, 1)   #cl
    np.place(y, y==16, 1)   #epi
    np.place(y, y==43, 1)   #vcl
    '''
    np.place(y, y!=1, 0)
    
    dic1[k] = x,y
#
dic_processing.pad_dic(dic1,max_len,max_len,0)
dic_processing.catogorate_dic(dic1,2)

x,y = dic_processing.toXY(dic1)
sample_num,max_len_sample,features_count = x.shape
assert max_len == max_len_sample


rnn_lay = GRU
model = Sequential()
#CNN
x = x.reshape(sample_num,max_len_sample,features_count,1)
#model.add(Reshape( (max_len,features_count,1),input_shape = (max_len,features_count)))
k = 7
f_c = 15
model.add(Conv2D(f_c,kernel_size = (1,7),activation = 'relu',input_shape = (max_len,features_count,1)))

model.add(Conv2D(f_c*2,kernel_size = (1,features_count-k+1),activation = 'relu'))
model.add(Reshape( (max_len,f_c*2)))
#rnn
model.add(Masking(mask_value=0))
#model.add(rnn_lay(30,input_dim = features_count, activation='tanh',return_sequences=True,implementation=1))
model.add(Bidirectional(rnn_lay(90,activation='tanh',return_sequences=True,implementation=1)))
model.add(rnn_lay(60,activation='tanh',return_sequences=True,implementation=1))
model.add(rnn_lay(30, activation='tanh',return_sequences=True,implementation=1))
model.add(TimeDistributed(Dense(2,activation='sigmoid')))

opt = SGD(lr = 0.001)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=4)
checkpoint = ModelCheckpoint(filepath = '../checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5')
model.fit(x,y,batch_size = 10,epochs = 200,callbacks=[early_stopping,checkpoint],validation_split = 0.05)

model.save('../models/sil_gru.model')
