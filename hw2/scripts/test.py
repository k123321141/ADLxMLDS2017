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

    d = myinput.load_x('../data/testing_data/feat/')
    '''
    model = my_model.model(80,4096,50,6528)
    model.load_weights('../models/5_val_0.92.cks')
    '''
    model = load_model('../models/9_val_0.87.cks')
    key = ['klteYv1Uv9A_27_33.avi','5YJaS2Eswg0_22_26.avi','UbmZAe5u5FI_132_141.avi','JntMAcTlOF0_50_70.avi','tJHUH9tpqPg_113_118.avi']
    for k in key:
        rowx = d[k]
        preds = my_model.my_pred(model,rowx,80,50)
        guess = decode(preds[0])
        print('key',k)
        print(guess.replace('<eos>',''))
        print('---')







