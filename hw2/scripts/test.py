# -*- coding: utf-8 -*-
from keras.callbacks import *
import keras
import numpy as np
import my_model
import myinput
import sys
from HW2_config import *
from keras.models import load_model
from myinput import decode,batch_decode
if __name__ == '__main__':
    assert len(sys.argv) == 3
    model_path = sys.argv[1]
    output_path = sys.argv[2]
    print('load testing data.')
    test_dic = myinput.load_x('../data/testing_data/feat/')
    print('load model.')
    model = load_model(model_path)
    #
    print('init decode map')
    vocab_map = myinput.init_vocabulary_map()
    decode_map = myinput.init_decode_map(vocab_map)
    #
    print('start prdiction.')
    with open(output_path,'w') as f:
        num = len(test_dic.keys())
        buf_x = np.zeros([num,input_len,feats_dim],dtype=np.float32)
        for i,k in enumerate(sorted(test_dic.keys())):
            buf_x[i,:,:] = test_dic[k]
        preds = my_model.batch_pred(model,buf_x,input_len,output_len)

        guess = batch_decode(decode_map,preds)
        for i,k in enumerate(sorted(test_dic.keys())):
            out = '%s,%s\n' % (k,guess[i].replace(' <eos>',''))
            f.write(out)
            print out
    print 'Done'




