import myinput
from keras.models import *
import numpy as np
import dic_processing

num_classes = 48
max_len = 777

#dic init setting,reshape

dic1 = myinput.load_test('../data/mfcc/test.ark')
x,fake_y = dic_processing.toXY(dic1)

sample_num,sample_len,features_count = x.shape

assert max_len == sample_len 

x = x.reshape(sample_num,max_len,features_count,1)

model = load_model('../models/sil.model')

y = model.predict(x)

keys = sorted(dic1.keys())
total = len(keys)

def split_test(y_seq):
    pre = y_seq[0]
    s = ''
    c = 1 if pre == 0 else 0
    for y in y_seq[1:]:
        if y == 1 and pre == 1:
            pre = y
        elif pre == 1 and y == 0:
            pre = y
            c = 1
        elif pre == 0 and y == 1:
            s += str(c)
            pre = y
        elif pre == 0 and y == 0:
            c += 1
            pre = y
        else:
            print 'error'
    return s

id = 'maeb0_si1411'
index = keys.index(id)
buf = y[index,:,:]
frame_seq = [argmax(buf[j,:]) for j in range(max_len)]
print split_test(frame_seq)


'''
for i in range(total):
    sentenceID = keys[i]
    buf = y[i,:,:]
    frame_seq = [argmax(buf[j,:]) for j in range(max_len)]

'''
