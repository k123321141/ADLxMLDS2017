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
dic_processing.pad_dic(dic1,max_len,max_len,num_classes)

dic_processing.catogorate_dic(dic1,num_classes+1)

x,y = dic_processing.toXY(dic1)
num = x.shape[0]




#model

x = x.reshape(num,max_len,features_count,1)

#sil_model = load_model('../models/sil_simple.model')
sil_model = load_model('../models/sil.model')
print 'prediction start'
z = sil_model.predict(x)
print 'prediction finished.'
print z.shape
num = z.shape[0]
x_slices = []
y_slices = []
buf = []
buf3 = []
buf4 = []
mini_len = 15
for i in range(num):
    sil_seq = [np.argmax(z[i,j,:]) for j in range(max_len)]
    #1 for sil
    start = sil_seq.index(0)
    sil_start = -1
    for j in range(start+1,max_len,1):
        if sil_seq[j] == 1 and sil_start == -1:      #start of sil slice 
            sil_start = j
        elif sil_seq[j] == 0 and sil_start != -1:    #end of sil slice 
            end = j
            if sil_start - start < mini_len or end - sil_start < 3 :     #sil slice is not long enough
                sil_start = -1
                continue

            #print 'start end',start,end,end-start
            x_slices.append( x[i,start:sil_start,:].reshape(sil_start-start,features_count) )
            y_slices.append( y[i,start:sil_start,:].reshape(sil_start-start,num_classes+1) )
            buf.append(sil_start-start)
            buf3.append(end - sil_start)
            start = end
            sil_start = -1
#find max length of slice
max_slice_len = 0
for slice_x in x_slices:
    max_slice_len = max_slice_len if max_slice_len > slice_x.shape[0] else slice_x.shape[0]
print 'max slice len : ' , max_slice_len
print 'total slice : ', len(x_slices)
buf2 = []
for i in range(num):
    sil_seq = [np.argmax(y[i,j,:]) for j in range(max_len)]
    #1 for sil
    for k in range(len(sil_seq)):
       if sil_seq[k] != 37 :
           start = k
           break
    sil_start = -1
    for j in range(start,max_len,1):
        if sil_seq[j] == 37 and sil_start == -1:      #start of sil slice 
            sil_start = j
        elif sil_seq[j] != 37 and sil_start != -1:    #end of sil slice 
            end = j
            if sil_start - start < mini_len:     #sil slice is not long enough
                sil_start = -1
                continue

            #print 'start end',start,end,end-start
            buf2.append(sil_start-start)
            buf4.append(end - sil_start)
            start = end
            sil_start = -1
import numpy as np
print 'size,max,min,mean,var :' ,len(buf),np.max(buf),np.min(buf),np.mean(buf),np.var(buf)
print 'size,max,min,mean,var :' ,len(buf2),np.max(buf2),np.min(buf2),np.mean(buf2),np.var(buf2)
print 'size,max,min,mean,var :' ,len(buf3),np.max(buf3),np.min(buf3),np.mean(buf3),np.var(buf3)
print 'size,max,min,mean,var :' ,len(buf4),np.max(buf4),np.min(buf4),np.mean(buf4),np.var(buf4)
print buf[0:10]
print buf2[0:10]
cnn_input = Input(shape=(max_len,features_count,1))
cnn_output = cnn_model(cnn_input)

seq_input = Input(shape=(max_len,num_classes+1))

#seq_output = auto_decoder_encoder.seq_output(cnn_output,max_out_len = max_out_len,num_classes = num_classes+1)
seq_output = auto_decoder_encoder.seq_output(seq_input,max_out_len = max_out_len,num_classes = num_classes+1)


model = Model(input = seq_input,output = seq_output)

plot_model(model, to_file='../model.png',show_shapes = True)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(z,y,batch_size = 10,epochs = 200,callbacks=[early_stopping],validation_split = 0.05)
print 'Done'
model.save('../models/seq2seq.model')
