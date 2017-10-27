import numpy as np
from keras.models import load_model

from mapping import * 
import myinput
import dic_processing
import random as r
import sys
#assert sentenceID & frameID follow the numeric order
#return sentence_dict
max_len = 777
num_classes = 48



def predict_output(model,test_path,output_path):
    
    dic_x = myinput.load_test(input_path)
    
    

    i = 1
    total = len(dic_x.keys())
    with open(output_path,'w') as f:
        f.write('id,phone_sequence\n')
        
        for setenceID in dic_x.keys():
            x = dic_x[setenceID]
            frame_len,feature_num = x.shape
#            padding for model
            x = np.pad(x,((0,max_len-frame_len),(0,0)),'constant', constant_values=0)
            x = x.reshape(1,max_len,feature_num,1)
            #
            y = model.predict(x)
            frame_seq = [np.argmax(y[0,j,:]) for j in range(y.shape[1])]

            s = convert_label_sequence(frame_seq).replace(',','')
         
            f.write('%s,%s\n' % (setenceID,s))
            
            print '%d/%d    %s,%s\n' % (i,total,setenceID,s)
            i+=1
def compare_output(model):
    
    
    
    #
    src = myinput.load_input('mfcc')
    keys = src.keys()
    r.shuffle(keys)
    sub_keys = keys[0:10]
    #sub_keys = keys
#    sub_keys = ['fadg0_si1279']
    sub_x ={k:(src[k]) for k in sub_keys }
    
    dic_x = sub_x

    i = 1
    total = len(dic_x.keys())
        
    for setenceID in dic_x.keys():
        x,src_y = dic_x[setenceID]
        frame_len,feature_num = x.shape
#            padding for model
        x = np.pad(x,((0,max_len-frame_len),(0,0)),'constant', constant_values=0)
        x = x.reshape(1,max_len,feature_num,1)
        #
        y = model.predict(x)
        frame_seq = [np.argmax(y[0,j,:]) for j in range(y.shape[1])]
    
        s = convert_label_sequence(frame_seq)
        

        src_y = src_y.reshape(frame_len)
        src_y = src_y.astype(np.int32)
        y_seq = src_y.tolist()
        s2 = convert_label_sequence(y_seq)[:-1]

        print '(%d/%d)  %s    \n' % (i,total,setenceID)
        h = 80
        k = int ( np.ceil((len(s) / float(h))) ) 
        for j in range(k):
            start = j*h
            end = (j+1)*h
            print 'result  :   %s\nsrc     :   %s\n' % (s[start:end],s2[start:end])
#        print 'result  :   %s\nsrc     :   %s\n' % (s[end:],s2[end:])

        i+=1
def generate_seq_y(output_path):
    
    dic_x = myinput.load_input('mfcc')
    
    i = 1
    total = len(dic_x.keys())
    with open(output_path,'w') as f:
         
        for setenceID in dic_x.keys():
            x,src_y = dic_x[setenceID]
            frame_len,feature_num = x.shape
            

            src_y = src_y.reshape(frame_len)
            src_y = src_y.astype(np.int32)
            y_seq = src_y.tolist()
            s2 = convert_label_sequence(y_seq)[:-1]

            print '(%d/%d)  %s    \n' % (i,total,setenceID)
            f.write('%s,%s\n' % (setenceID,s2))
            i+=1
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
    


    print 'Done'




