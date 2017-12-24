# -*- coding: utf8 -*-
import numpy as np
from os.path import join
from scipy.ndimage import imread
from scipy.misc import imresize, imsave
tags = ['hair ','eyes ','green ','red ',' hair',' eyes',' green',' red']
import json
def main():
   
    p = './tags_clean.csv'
    with open(p,'r') as f:
        ls = f.readlines()
    s = set()
    x_buf = []
    y_buf = []
    dic = {}
    for l in ls:
        if tag_in(l): 
            buf = l.split(',')
            idx = buf[0]
            ll = buf[-1].strip().split('\t')
            for lll in ll:
                t = lll.split(':')[0]
                if tag_in(t) and '1' not in t:
                    img_path = join('./faces',idx+'.jpg')
                    x = imread(img_path)
                    x = imresize(x, [64,64,3])
                    x = x.reshape([1,64,64,3])
                    y = encode(t)
                    x_buf.append(x)
                    y_buf.append(y)
                    s.add(t)
    print s
    x = [len(t) for t in s]
    print max(x)
    '''
    #tags = ['purple eyes','aqua eyes','green eyes','black hair','brown eyes','blue eyes','blonde hair',\
            'green hair','black eyes','pink hair','blue hair','brown hair']
    '''
    #dic = []
    x,y = (np.vstack(x_buf), np.vstack(y_buf))
    with open('./x','wb') as f:
        np.save(f,x)
    with open('./y','wb') as f:
        np.save(f,y)

def main2():
   
    p = './tags_clean.csv'
    with open(p,'r') as f:
        ls = f.readlines()
    dic = {}
    for l in ls:
        if tag_in(l): 
            buf = l.split(',')
            idx = buf[0]
            ll = buf[-1].strip().split('\t')
            for lll in ll:
                t = lll.split(':')[0]
                if tag_in(t) and '1' not in t:
                    img_path = join('./faces',idx+'.jpg')
                    if 'red' in t:
                        dic[int(idx)] = 'red'
                    elif 'green' in t:
                        dic[int(idx)] = 'green'
                    break
    with open('./color.json','w') as f:
        s = json.dumps(dic)
        f.write(s)
def tag_in(s):
    for tag in tags:
        if tag in s:
            return True
    return False
def encode(s):
    y = np.zeros([1,32,27])
    for idx,c in enumerate(s.lower()):
        i = ord(c) - 97 if c != ' ' else 26
        if i<0 or i > 26:
            print s,c
        y[0,idx,i] = 1
    return y
def read_x():
    with open('./x.npy') as f:
        x = np.load(f)
    print x.shape
if __name__ == '__main__':
    main2()
    #read_x()
