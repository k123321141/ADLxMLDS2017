from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.utils import plot_model
import myinput
import numpy as np
import dic_processing
import cnn 
import auto_decoder_encoder

max_len = 777
num_classes = 48
features_count = 39
max_out_len = 80
seq = True

#dic init setting,reshape
dic1 = myinput.load_input('mfcc')
if seq == True:
    seq_dict = myinput.read_seq_Y('../data/train.lab')
    for k in dic1.keys():
        x,y = dic1[k]
        seq_y = seq_dict[k]
        dic1[k] = x,seq_y
    dic_processing.pad_dic(dic1,max_len,max_out_len,num_classes)
else:
    dic_processing.pad_dic(dic1,max_len,max_len,num_classes)

dic_processing.catogorate_dic(dic1,num_classes+1)

x,y = dic_processing.toXY(dic1)
num = x.shape[0]




#model

x = x.reshape(num,max_len,features_count,1)

cnn_input = Input(shape=(max_len,features_count,1))
cnn_output = cnn.cnn_output(cnn_input)
seq_output = auto_decoder_encoder.seq_output(cnn_output,max_out_len = max_out_len,num_classes = num_classes+1)


model = Model(input = cnn_input,output = seq_output)

plot_model(model, to_file='../model.png',show_shapes = True)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x,y,batch_size = 100,epochs = 200,callbacks=[early_stopping],validation_split = 0.05)
print 'Done'
model.save('../models/seq2seq.model')
