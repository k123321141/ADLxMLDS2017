# -*- coding: utf8 -*-
import numpy as np
from os.path import join
from scipy.ndimage import imread
from scipy.misc import imresize, imsave
import json, random

colors = ['green', 'white', 'blue', 'aqua', 'gray', 'purple', 'red', 'pink', 'yellow', 'brown', 'black']
parts = ['hair', 'eye']

def main():
   
    p = './tags_clean.csv'
    with open(p,'r') as f:
        ls = f.readlines()
    tag_set = set()
    x_buf = []
    y_buf = []
    wrong_buf = []
    for line in ls:
        idx, line = line.split(',')
        tags = line.strip().split('\t')
        s = ''
        for tag in tags:
            tag = tag.split(':')[0]
            if tag_in(tag, colors) and tag_in(tag, parts):
                s += tag + ','
        if s != '':
            s = s[:-1]
            y = encode(s)
            img_path = join('./faces',idx+'.jpg')
            x = imread(img_path)
            x = imresize(x, [64,64,3])
            x = x.reshape([1,64,64,3]) / 255
            x_buf.append(x)
            y_buf.append(y)
            wrong_y = wrong_text(y)
            wrong_buf.append(wrong_y)
            #print idx,s
            #tag_set.add(s)
    x, y, wrong_y = (np.vstack(x_buf), np.vstack(y_buf), np.vstack(wrong_buf) ) 

    #print tag_set, len(tag_set)
    with open('./train.npz','wb') as f:
        np.savez(f, x=x, y=y, wrong_y=wrong_y)
    print x.shape, y.shape, wrong_y.shape
def wrong_text(y):
    shape = (1, len(colors) * len(parts))
    assert y.shape == shape
    arr = []
    for i in range(y.shape[-1]):
        if y[0,i] == 0:
            arr.append(i)
    wrong_y = np.zeros(shape)
    for i in random.sample(arr, 4):
        wrong_y[0,i] = 1
    return wrong_y

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
def tag_in(s, tags):
    for tag in tags:
        if tag in s:
            return True
    return False
def encode(s):
    y = np.zeros([1, len(colors) * len(parts)])
    #example purple eyes, green hair
    tags = s.split(',')
    for tag in tags:
        for i,part in enumerate(parts):

            #eyes in green eyes
            if part in tag:
                idx = -1
                for j,color in enumerate(colors):
                    if color in tag:
                        idx = j
                        break
                assert idx != -1
                y[0,i*len(parts) + idx] = 1
    return y

def read_x():
    with open('./x.npy') as f:
        x = np.load(f)
    print x.shape
if __name__ == '__main__':
    main()
    #read_x()
