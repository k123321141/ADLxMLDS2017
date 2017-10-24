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
    np.place(y, y!=1, 0)
    
    dic1[k] = x,y
#
dic_processing.pad_dic(dic1,max_len,max_len,0)
dic_processing.catogorate_dic(dic1,2)

x,y = dic_processing.toXY(dic1)
sample_num,max_len_sample,features_count = x.shape


model = Sequential()
model.add(Masking(input_shape = (777,39)))
model.add(TimeDistributed(Dense(200,activation='tanh',input_shape = (777,39))))
model.add(TimeDistributed(Dense(200,activation='tanh')))
model.add(TimeDistributed(Dense(200,activation='tanh')))
model.add(TimeDistributed(Dense(200,activation='tanh')))
model.add(TimeDistributed(Dense(200,activation='tanh')))
model.add(TimeDistributed(Dense(200,activation='tanh')))
model.add(TimeDistributed(Dense(200,activation='tanh')))
model.add(TimeDistributed(Dense(200,activation='tanh')))

model.add(Dense(2,activation='sigmoid'))


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x,y,batch_size = 400,epochs = 200,callbacks=[early_stopping,],validation_split = 0.05)

model.save('../models/sil_simple.model')
