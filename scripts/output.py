import numpy as np
from keras.models import *
import mapping
import myinput
#assert sentenceID & frameID follow the numeric order
#return sentence_dict
max_len = 777
features_count = 39
num_classes = 48


def predict_output(model,sentence_dict,output_path,map_48_int_dict,map_48_reverse,map_48_char_dict,map_48_39_dict):

    x = [sentence_dict[k][0] for k in sorted(sentence_dict.keys())]
    
    y = model.predict(x)

    #for counting
    total = len(sentence_dict.keys())
    index = 1
    with open(output_path,'w') as f:
        f.write('id,phone_sequence\n')
        for sentence_id in sentence_dict.keys():
            x = sentence_dict[sentence_id]
            y_arr = list( ([np.argmax(i, axis=1) for i in model.predict(x)])[0])


            s = mapping(y_arr,rev_dic,map_48_int_dict,map_48_char_dict,map_48_39_dict)
            
            f.write('%s,%s\n' % (sentence_id,s))
            print '%d/%d    %s,%s\n' % (index,total,sentence_id,s)
            index += 1
#y is a list
def mapping(y_arr,rev_dic,map_48_int_dict,map_48_char_dict,map_48_39_dict):
    if num_classes in y_arr:
        y_arr = y_arr[0: y_arr.index(num_classes)]
    c_arr = [rev_dic[y] for y in y_arr]
        
            
    c_arr = [map_48_39_dict[c] for c in c_arr]
    c_arr = trim_arr(c_arr)
    c_arr = [map_48_char_dict[c] for c in c_arr]

    s = ''
    for c in c_arr:
        s += c
    return s



def trim_arr(arr):
    #trim sil
    for i in range(len(arr)):
        a = arr[i]
        if a != 'sil':
            start_index = i
            break
    arr = arr[start_index:]

    for i in range(len(arr)-1,-1,-1):
        a = arr[i]
        if a != 'sil':
            end_index = i
            break
    arr = arr[:end_index+1]

    #remove repeat
    pre = 'start'
    for i in range(len(arr)):
        a = arr[i]
        if a == pre:
            arr[i] = 'repeat'
        else:
            pre = a
    #
    result = []
    for a in arr:
        if a != 'repeat':
            result.append(a)
    return result


def map48_39(map_file_path):

    #read map file
    with open(map_file_path,'r') as f:
        lines = f.readlines()
    result = {}
    for l in lines:
        l = l.strip()
        src,dst = l.split('\t')
        result[src] = dst
    return result
def reverse_dic(dic):
    inv_map = {v: k for k, v in dic.iteritems()}
    return inv_map
def map_phone_char(map_file_path,to_char = False):
    #read map file
    with open(map_file_path,'r') as f:
        lines = f.readlines()
    result = {}
    for l in lines:
        l = l.strip()
        src,dst_index,dst_char = l.split('\t')
        if to_char:
            result[src] = dst_char
        else:
            result[src] = int(dst_index)

    return result


#read and save
if __name__ == '__main__':
    output_path = '../data/output.csv'
    model_path = '../models/seq2seq.model'
    test1_path = '../data/mfcc/test.ark'
    test2_path = '../data/fbank/test.ark'
    
    dic1 = read_X(test1_path)
    dic2 = read_X(test2_path)
    
    
    
    map_48_int_dict,map_48_reverse,map_48_char_dict,map_48_39_dict = mapping.read_maps()
    
    sentence_dict = myinput.read_X(data_path)
    model = load_model(model_path)
    
    predict_output(model,sentence_dict,output_path,map_48_int_dict,map_48_reverse,map_48_char_dict,map_48_39_dict)

    print 'Done'
#read npz
def load_input():
    npz_path = './bin.npz'

    return read_npz(npz_path)





