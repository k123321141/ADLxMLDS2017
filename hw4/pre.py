# -*- coding: utf8 -*-
import numpy as np
from os.path import join
from scipy.ndimage import imread
from scipy.misc import imresize, imsave
import json, random

colors = ['<unk>','green', 'white', 'blue', 'aqua', 'gray', 'purple', 'red', 'pink', 'yellow', 'brown', 'black']
parts = ['eyes', 'hair']

def main():
   
    p = './tags_clean.csv'
    with open(p,'r') as f:
        ls = f.readlines()
    tag_set = set()
    x_buf = []
    y1_buf = []
    y2_buf = []
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
            #print s
            y1,y2 = encode(s)
            print s,y1,y2
            img_path = join('./faces',idx+'.jpg')
            x = imread(img_path)
            x = imresize(x, [64,64,3]).astype(np.uint8)
            x = x.reshape([1,64,64,3]) 
            x_buf.append(x)
            y1_buf.append(y1)
            y2_buf.append(y2)
            #print idx,s
            #tag_set.add(s)
    x, y1, y2  = (np.vstack(x_buf), np.vstack(y1_buf), np.vstack(y2_buf)) 

    #print tag_set, len(tag_set)
    with open('./train.npz','wb') as f:
        np.savez(f, x=x, eyes=y1, hair=y2)
    print x.shape, y1.shape, y2.shape
def tag_in(s, tags):
    for tag in tags:
        if tag in s:
            return True
    return False
def encode(s):
    y = [np.zeros([1, 1]) for i in range(len(parts))]
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
                y[i][0,0] = idx
    return y
"""
def encode_pre(s):
    y = [np.zeros([1, len(colors)]) for i in range(len(parts))]
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
                y[i][0,idx] = 1
    return y
 """

def read_x():
    with open('./x.npy') as f:
        x = np.load(f)
    print x.shape
if __name__ == '__main__':
    main()
    #read_x()
