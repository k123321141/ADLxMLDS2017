# -*- coding: utf-8 -*-
from keras.callbacks import *
import keras
import numpy as np
import utils
import sys
import myinput
import HW2_config
import custom_recurrents
from keras.models import load_model

from myinput import decode,batch_decode

vocab_map = myinput.init_vocabulary_map()
decode_map = {vocab_map[k]:k for k in vocab_map.keys()}

def main(model,data_path,video_name_list,output_path):
    print('load testing data.')
    #test_dic = myinput.load_x_dic('../data/testing_data/feat/')
    test_dic = myinput.load_test_dic(data_path,video_name_list)
    #
    print('start prdiction.')
    with open(output_path,'w') as f:
        num = len(test_dic.keys())
        buf_x = np.zeros([num,HW2_config.input_len,HW2_config.input_dim],dtype=np.float32)
        for i,k in enumerate(sorted(test_dic.keys())):
            buf_x[i,:,:] = test_dic[k]
        preds = utils.batch_pred(model,buf_x,HW2_config.output_len)

        guess = batch_decode(decode_map,preds)
        for i,k in enumerate(sorted(test_dic.keys())):
            out = '%s,%s\n' % (k,guess[i].replace(' <eos>',''))
            f.write(out)
    print('Done')
if __name__ == '__main__':
    assert len(sys.argv) == 4
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    output_path = sys.argv[3]
    #../data/testing_data/feat/
    fl = os.listdir(data_path)
    file_name_list = [f for f in fl if f.endswith('.npy')]

    model = load_model(model_path,
                custom_objects={'loss_with_mask':utils.loss_with_mask,
                    'acc_with_mask':utils.acc_with_mask,
                    'AttentionDecoder':custom_recurrents.AttentionDecoder})
    main(model,data_path,file_name_list,output_path)


