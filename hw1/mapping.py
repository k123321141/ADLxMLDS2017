from os.path import join

def init(data_dir):
    map_48_int_dict,map_48_reverse,map_48_char_dict,map_48_39_dict = read_maps(join(data_dir,'phones','48_39.map'),join(data_dir,'48phone_char.map'))


def read_maps(mapfile_48_39 = '../data/48_39.map',mapfile_phone_char = '../data/48phone_char.map'):

    map_48_int_dict = map_phone_char(mapfile_phone_char,to_char=False)
    map_48_reverse = { v:k for (k,v) in map_48_int_dict.items() }
    map_48_char_dict = map_phone_char(mapfile_phone_char,to_char=True)
    map_48_39_dict = map48_39(mapfile_48_39)

    return map_48_int_dict,map_48_reverse,map_48_char_dict,map_48_39_dict



def mapping(y_list,map_name):

    if map_name == '48_int':
        map_dic = map_48_int_dict
    elif map_name == '48_reverse':
        map_dic = map_48_reverse
    elif map_name == '48_char':
        map_dic = map_48_char_dict
    elif map_name == '48_39':
        map_dic = map_48_39_dict
    else:
        map_dic = None

    return [map_dic[y] for y in y_list]

def trim_sil(arr):
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
    
    return arr

def trim_repeat(arr):
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
