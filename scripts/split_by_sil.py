from keras.models import *
from keras.layers import *
from keras.preprocessing.sequence import *
from keras.callbacks import *
from keras.utils import plot_model
import myinput
import numpy as np
import dic_processing
import cnn 
import auto_decoder_encoder

def split_by_sil(x,y):
    num_classes = 48
    features_count = 39
    max_len = 777
#x = x.reshape(num,max_len,features_count,1)

    sil_model = load_model('../models/sil_simple.model')
#sil_model = load_model('../models/sil.model')
    print 'prediction start'
    z = sil_model.predict(x)
    print 'prediction finished.'


    num = z.shape[0]
    x_slices = []
    y_slices = []
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
                if sil_start - start < mini_len :     #sil slice is not long enough
                    sil_start = -1
                    continue
                #contain the sil at tail 
                x_slices.append( x[i,start:sil_start,:].reshape(sil_start-start,features_count) )
                y_slices.append( y[i,start:sil_start,:].reshape(sil_start-start,num_classes+1) )
                start = end
                sil_start = -1
#find max length of slice
    max_slice_len = 0
    for slice_x in x_slices:
        max_slice_len = max_slice_len if max_slice_len > slice_x.shape[0] else slice_x.shape[0]
    print 'max slice len : ' , max_slice_len
    print 'total slice : ', len(x_slices)

    padding_len = max_slice_len + 10

    x = pad_sequences(x_slices,padding ='post',maxlen = padding_len)
    y = pad_sequences(y_slices,padding ='post',maxlen = padding_len,value = num_classes)

    print 'split by sil to %d slice, x = %s ' % (len(x_slices),x.shape)

    return padding_len,x,y 

