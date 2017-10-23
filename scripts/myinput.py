import numpy as np
import sys
import mapping
#return a dict which contain
#keys = {maeb0_si1411,maeb0_si2250,...} ->  sentenceID  ->  string
#vals = {(x1,y1),(x2,y2),...}           ->  tuple       ->  x=ndarray(num,39),y=ndarray(num,1)


def read_X(data_path):
    sentence_dict = {}
    print 'read X ',data_path
    #training data
    with open(data_path,'r') as f:
        lines = f.readlines()
    for l in lines:
        l = l.strip().split(' ')
        #fadg0_si1279_3
        speaker,sentence,frame = l[0].split('_')
        #fadg0_si1279
        key = speaker + '_' + sentence
        
        vals = [ float(x) for x in l[1:] ]
        if sentence_dict.has_key(key) != True:
            sentence_dict[key] = {}
        frame_dict = sentence_dict[key]
        frame_dict[int(frame)] = vals


    return sentence_dict

#return dict[]
def read_Y(label_path,map_48_int_dict,map2int = True):
    sentence_dict = {}
    #label
    with open(label_path,'r') as f:
        lines = f.readlines()
    for l in lines:
        buf,label = l.strip().split(',')
        
        #fadg0_si1279_3
        speaker,sentence,frame = buf.split('_')
        #fadg0_si1279
        key = speaker + '_' + sentence

        #mapping
        #from char to int,sil -> 37,range[0,47]
        if map2int:
            label = map_48_int_dict[label]
        
        #new sentence during processing
        if sentence_dict.has_key(key) != True:
            sentence_dict[key] = {}
        #set y to specied frame id
        frame_dict = sentence_dict[key]
        
        frame_dict[int(frame)] = label
    return sentence_dict

def read_seq_Y(label_path):
    sentence_dict = {}
    #label
    with open(label_path,'r') as f:
        lines = f.readlines()
    for l in lines:
        buf = l.strip().split(',')
        key = buf[0]
        labels = buf[1:]
        
        labels = mapping.mapping(labels,'48_int')
        
 
        for i in range(len(labels)):
            label = labels[i]
            if sentence_dict.has_key(key) != True:
                sentence_dict[key] = {}
            frame_dict = sentence_dict[key]
            
            
            frame_dict[i+1] = label
        sentence_dict[key] = frame_dict

    return sentence_dict

#return a dict which paired with ,key:sentence ID,val:(x,y) np.array
def combine(X,Y):
    dic = {}
    assert sorted(X.keys()) == sorted(Y.keys())
    for sentenceID in np.sort(X.keys()):
        X_dict = X[sentenceID]
        Y_dict = Y[sentenceID]
        assert sorted(X_dict.keys()) == sorted(Y_dict.keys())
        x = dic2ndarray(X_dict)
        y = dic2ndarray(Y_dict)
        y = y.reshape(y.shape[0],1)

        
        dic[sentenceID] = (x,y)
    return dic
def dic2ndarray(dic):
    buf = []
    #sort frame ID
    for i in sorted(dic.keys()):
        buf.append(dic[i])
    
    return np.asarray(buf,dtype=np.float32)


def read_npz(path):
    print 'reading ' , path
    
    loaded = np.load(path)
    buf = loaded['a']
    dic = buf.item()
    print 'reading done'

    return dic
def write_npz(dic,path):
    print 'saving ' , path
    np.savez_compressed(path,a=dic)
    print 'saving done.'

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
def stack_x(dic1,dic2):
    dic3 = {}
    assert sorted(dic1.keys()) == sorted(dic2.keys())
    buf_x = []
    for sentenID in dic1.keys():
        x,y = dic1[sentenID]
        x2,y2 = dic2[sentenID]
        x3 = np.hstack([x,x2])
        dic3[sentenID] = (x3,y)

    return dic3
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
def init_npz():
    mfcc_path = '../data/mfcc/train.ark'
    fbank_path = '../data/fbank/train.ark'
    test_path = '../data/test.ark'
    label_path = '../data/train.lab'
    npz_path = '../data/bin.npz'
    mapfile_48_39 = '../data/48_39.map'
    mapfile_phone_char = '../data/48phone_char.map'
    
    map_48_int_dict = map_phone_char(mapfile_phone_char,to_char=False)
#    map_48_39_dict = map48_39(mapfile_48_39)

    y_dic = read_Y(label_path,map_48_int_dict)
    mfcc_dic = combine(read_X(mfcc_path),y_dic)
    fbank_dic = combine(read_X(fbank_path),y_dic)
    
    write_npz(mfcc_dic,'../data/mfcc.npz')
    write_npz(fbank_dic,'../data/fbank.npz')
    
#read npz
def load_input(feature_name = 'mfcc'):

    if feature_name == 'mfcc':
        npz_path = '../data/mfcc.npz'
    elif feature_name == 'fbank':
        npz_path = '../data/fbank.npz'
    else:
        print 'error'
        sys.exit(1)

    return read_npz(npz_path)

#return a dict,key : sentenceID ,vals : ndarray in shape(len,feature_number)
def load_test(path):

    x = read_X(path)
    for setenceID in x.keys():
        buf_x = []
        frame_dic = x[setenceID]
        for index in sorted(frame_dic.keys()):
            buf_x.append(frame_dic[index])
        x[setenceID] = np.asarray(buf_x)
    
    return x
if __name__ == '__main__':
    test()







