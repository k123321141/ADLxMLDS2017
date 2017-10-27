import numpy as np
from keras.models import load_model

from mapping import * 
import myinput
import dic_processing
import random as r
import sys
from configuration import max_len,num_classes
from keras.utils import to_categorical

#assert sentenceID & frameID follow the numeric order
#return sentence_dict
max_len = 777
num_classes = 48



def predict_output(model,test_path,output_path):
    
    sentence_dict = read_data(test_path)

    dic_processing.pad_dic(sentence_dict,max_len,0)
    x = dic_processing.vstack_dic(sentence_dict)
    
    
    print('start predict..')
    y = model.predict(x)
    print('output prediction to %s ..' % output_path)
   
    #
    keys = sorted(sentence_dict.keys())
    #

    with open(output_path,'w') as f:
        f.write('id,phone_sequence\n')
        
        for i in range(len(keys)):
            sentenceID = keys[i]

            frame_seq = [np.argmax(y[i,j,:]) for j in range(max_len)]

            s = convert_label_sequence(frame_seq).replace(',','')
         
            f.write('%s,%s\n' % (setenceID,s))
            
def compare_output(model):
    
    
    
    #
    x,y = myinput.load_input('mfcc')
    #random select sample
    sample_count = 5
    random_indices = range(x.shape[0])
    r.shuffle(random_indices)
    random_indices = random_indices[:sample_count]
    #
    sub_x = x[random_indices,:,:]
    sub_y = y[random_indices,:,:]
    #
    
    
    for i in range(sample_count):
        
        x = sub_x[i,:,:]
        y_true = sub_y[i,:,:]
        #
        y_pred = model.predict(x)
    
        y_true_seq = [np.argmax(y_true[j,:]) for j in range(y_true.shape[0])]  #777
        y_pred_seq = [np.argmax(y_pred[j,:]) for j in range(y_pred.shape[0])]
    
        s_true = convert_label_sequence(y_true_seq)
        s_pred = convert_label_sequence(y_pred_seq)
        

        print( '(%d/%d)    \n' % (i+1,sample_count) )
        h = 80
        k = int ( np.ceil((len(s) / float(h))) ) 
        for j in range(k):
            start = j*h
            end = (j+1)*h
            print ( 'result  :   %s\nsrc     :   %s\n' % (s_pred[start:end],s_true[start:end]) )
#        print( 'result  :   %s\nsrc     :   %s\n' % (s_pred[end:],s_true[end:]))

#assert input y is one-hot and with padding tensor
def to_seq_y(y_with_padding,max_len=80):
    y = y_with_padding
    assert max_len < y.shape[1]
    buf = []
    for i in range(y.shape[0]):
        y_true_seq = [np.argmax(y_true[i,j,:]) for j in range(y.shape[1])]  #777
        #eos trimming
        eos = (y.shape[-1] - 1)
        if eos in y_true_seq:
            y_true_seq = y_true_seq[0: y_true_seq.index(eos)]
        c_arr = y_true_seq
        
        #reversed from index to char
        c_arr = mapping(c_arr,'48_reverse')
            
        #48 -> 39   
        c_arr = mapping(c_arr,'48_39')
       
        #trimming
        c_arr = trim_sil(c_arr)
        c_arr = trim_repeat(c_arr)

        #back to index
        c_arr = mapping(c_arr,'48_int')
        #padding eos
        if(len(c_arr) < max_len):
            c_arr.extend( [eos]*( max_len-len(c_arr) ) )
        #to ndarray
        y_ndarr = np.asarray(c_arr).reshape(1,max_len)
        #to one hot
        y_ndarr = to_categorical(y_ndarr,eos+1)
        buf.append(y_ndarr)
    return np.vstack(buf)
#y is a list
def convert_label_sequence(label_seq):
    #eof trimming
    if num_classes in label_seq:
        label_seq = label_seq[0: label_seq.index(num_classes)]
    c_arr = label_seq
    
    #reversed from index to char
    c_arr = mapping(c_arr,'48_reverse')
        
    #48 -> 39   
    c_arr = mapping(c_arr,'48_39')
   
    #trimming
    #c_arr = trim_sil(c_arr)
    #c_arr = trim_repeat(c_arr)

    #index -> char
    c_arr = mapping(c_arr,'48_char')

    #to string
    s = ''
    for c in c_arr:
        s += c +','
    return s

#python <model_path>   ->   output random 10 training predict with correct labels to stdout
#python <model_path> <test_path> <output_path> ->   output test predict to <output_path>
if __name__ == '__main__':
    '''
    output_path = '../data/output.csv'
    model_path = '../models/seq.13-1.60.hdf5'
    test_path = '../data/mfcc/test.ark'
    '''
    assert len(sys.argv) == 2 or len(sys.argv) == 4
    model_path = sys.argv[1]
    model = load_model(model_path)
    
    if len(sys.argv) == 2:
        compare_output(model)
    else:
        test_path = sys.argv[2]
        output_path = sys.argv[3]

        predict_output(model,test_path = test_path,output_path = output_path)
    


    print('Done')




