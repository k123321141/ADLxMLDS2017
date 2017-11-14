# -*- coding: utf-8 -*-
from keras.callbacks import *
import keras
import numpy as np
import my_model
import myinput
import sys
from keras.models import load_model
vocab_map = myinput.init_vocabulary_map()
decode_map = {vocab_map[k]:k for k in vocab_map.keys()}
def batch_decode(y):
    num,output_len = y.shape[0:2]
    y = np.argmax(y,axis = -1)
    output_list= []
    for i in range(num):
        s = ''
        for j in range(output_len):
            vocab_idx = y[i,j]
            if vocab_idx != 0:      #<pad>
                s += decode_map[vocab_idx] + ' '
        s = s.strip() + '.' 
        output_list.append(s.encode('utf-8'))
    return output_list
def decode(y):
    output_len = y.shape[0]
    y = np.argmax(y,axis = -1)
    s = ''
    for j in range(output_len):
        vocab_idx = y[j]
        if vocab_idx != 0:      #<pad>
            s += decode_map[vocab_idx] + ' '
    s = s.strip() + '.' 
    return s.encode('utf-8')
if __name__ == '__main__':
    assert len(sys.argv) == 3
    model_path = sys.argv[1]
    output_path = sys.argv[2]
    print('load testing data.')
    test_dic = myinput.load_x('../data/testing_data/feat/')
    print('load model.')
    model = load_model(model_path)
    print('start prdiction.')
    with open(output_path,'w') as f:
        for k in test_dic.keys():
            #(1,80,4096)
            rowx = test_dic[k]
            print rowx.shape
            sys.exit(0)
            preds = my_model.my_pred(model,rowx,80,50)
            guess = decode(preds[0]).replace(' <eos>','')
        
            out = '%s,%s\n' % (k,guess)
            f.write(out)
            print out

    print 'Done'




