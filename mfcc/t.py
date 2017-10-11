import numpy as np

#assert sentenceID & frameID follow the numeric order
def read_input(data_path,label_path,map_48_39_dict,map_48_int_dict):
    sentence_dict = {}
    
    #training data
    with open(data_path,'r') as f:
        lines = f.readlines()
    for l in lines:
        l = l.strip().split(' ')
        speaker,sentence,frame = l[0].split('_')
        key = speaker + '_' + sentence
        
        vals = [ float(x) for x in l[1:] ]
        if sentence_dict.has_key(key) != Ture:
            sentence_dict[key] = {}
        frane_dict = sentence_dict[key]
        frane_dict[int(frame)] = vals

    #label
    with open(label_path,'r') as f:
        lines = f.readlines()
    for l in lines:
        l = l.strip().split(',')
        key,label = l
        
        #mapping
        label = map_48_39_dict[label]
        label = map_48_int_dict[label]
        dict[key].append(label)

    print 'numpy'
    #convert to numpy format
    buf = np.asarray(dict.values(),dtype = np.float32)
    x,y = np.split(buf,[-1],axis =1) #the last column is y
    
    return x,y
def read_npz(path):
    loaded = np.load(path)
    x = loaded['a']
    y = loaded['b']

    return x,y
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


#assert sentenceID & frameID follow the numeric order
def read_input2(path):
    dict = {}
    with open(path,'r') as f:
        lines = f.readlines()
    for l in lines:
        l = l.strip().split(' ')
        speaker,sentence,frame = l[0].split('_')
        
        vals = [ float(x) for x in l[1:] ]
        key = speaker + '_' + sentence
        
        if dict.has_key(key) != True:
            dict[key] = []
        dict[key].append(vals)
    #dict


    return dict

#read and save
def test():
    data_path = './train.ark'
    label_path = '../train.lab'
    npz_path = './bin.npz'
    mapfile_48_39 = '../48_39.map'
    mapfile_phone_char = '../48phone_char.map'
    
    map_48_int_dict = map_phone_char(mapfile_phone_char,to_char=False)
    map_48_39_dict = map48_39(mapfile_48_39)
    
    
    x,y = read_input(data_path,label_path,map_48_39_dict,map_48_int_dict)
    
    print 'saving.. ' , npz_path
    np.savez_compressed(npz_path,a=x,b=y)

    return x,y
#read npz
def test2():
    npz_path = './bin.npz'
    x,y = read_npz(npz_path)

    return x,y





