import numpy as np

#return a dict which contain
#keys = {maeb0_si1411,maeb0_si2250,...} ->  sentenceID  ->  string
#vals = {(x1,y1),(x2,y2),...}           ->  tuple       ->  x=ndarray(num,39),y=ndarray(num,1)


def read_X(data_path):
    sentence_dict = {}
    
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
def read_Y(label_path,map_48_int_dict):
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
        label = map_48_int_dict[label]
        
        #new sentence during processing
        if sentence_dict.has_key(key) != True:
            sentence_dict[key] = {}
        #set y to specied frame id
        frame_dict = sentence_dict[key]
        
        frame_dict[int(frame)] = label
    return sentence_dict
#return a dict which paired with ,key:sentence ID,val:(x,y) np.array
def combine(X,Y):
    dic = {}
    assert X.keys() == Y.keys()
    for sentenceID in np.sort(X.keys()):
        X_dict = X[sentenceID]
        Y_dict = Y[sentenceID]
        assert X.dict.keys() == Y.dict.keys()
        
        buf_x = []
        buf_y = []
        #sort frame ID
        for i in np.sort(X_dict.keys()):
            buf_x.append(X_dict[i])
            buf_y.append(Y_dict[i])
        
        x = np.asarray(buf_x,dtype=np.float32)
        y = np.asarray(buf_y,dtype=np.float32)
        

        dic[sentenceID] = (x,y)
    return dic


def read_npz(path):
    print 'reading ' , path
    
    loaded = np.load(path)
    buf = loaded['a']
    dic = buf.item()
    print 'reading done'

    return dic
def write_npz(dic,path):
    print 'saving ' , path
    np.savez_compressed(npz_path,a=dic)
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
def test():
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
    mfcc_dic = combine(read_X(mfcc_path,y_dic)
    fbank_dic = combine(read_X(fbank_path,y_dic)
    
    write_npz(mfcc_dic,'../data/mfcc.npz')
    write_npz(fbank_dic,'../data/fbank.npz')
    
#read npz
def load_input(npz_path = 'mfcc'):
    if npz_path == 'mfcc':
        npz_path = '../data/mfcc.npz'
    else:
        npz_path = '../data/fbank.npz'

    return read_npz(npz_path)
if __name__ == '__main__':
    test()





