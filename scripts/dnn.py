import myinput
from keras.models import *
from keras.layers.recurrent import SimpleRNN
from keras.layers import *
from keras.utils import *
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
import numpy as np
import random

num_classes = 48

dic1 = myinput.load_input('mfcc')
dic2 = myinput.load_input('fbank')
dic3 = myinput.stack_x(dic1,dic2)
buf_x = []
buf_y = []
for sentenID in dic3.keys():
    x,y = dic3[sentenID]
    buf_x.append(x)
    buf_y.append(y)

X = np.vstack(buf_x)
Y = np.vstack(buf_y)

Y = to_categorical(Y,num_classes)

print X.shape,Y.shape

features_count = X.shape[1]

model = Sequential()

model.add(Dense(features_count,input_shape = (features_count,),activation = 'sigmoid'))
model.add(Dense(features_count,activation = 'sigmoid'))
model.add(Dense(features_count,activation = 'sigmoid'))
model.add(Dense(features_count,activation = 'sigmoid'))
model.add(Dense(features_count,activation = 'sigmoid'))
model.add(Dense(num_classes,activation = 'softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#plot_model(model, to_file='../../model.png')
#training loop
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(X,Y,epochs = 200,batch_size = 1000,validation_split = 0.05,callbacks = [early_stopping])

model.save('./dnn.model')

