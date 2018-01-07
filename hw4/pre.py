# -*- coding: utf8 -*-
import numpy as np
from os.path import join
from scipy.ndimage import imread
from scipy.misc import imresize, imsave
import json, random, re

random.seed(0413)

colors = ['<unk>','green', 'white', 'blue', 'aqua', 'gray', 'purple', 'red', 'pink', 'yellow', 'brown', 'black']
parts = ['eyes', 'hair']

#configure eyes pattern
patt = ''
for c in colors:
    patt += c + ' eyes|'
patt = patt[:-1]
eyes_patten = re.compile(patt)

#configure hair pattern
patt = ''
for c in colors:
    patt += c + ' hair|'
patt = patt[:-1]
hair_patten = re.compile(patt)

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
        tags = line.strip().lower()
        tags = tags.replace('eye','eyes')

        #eyes
        e_m = eyes_patten.findall(tags)
        if len(e_m) == 0:
            y1 = 0
        else:
            idxs = [colors.index( s.replace(' eyes','') ) for s in e_m]
            y1 = random.choice(idxs)
        #hair
        h_m = hair_patten.findall(tags)
        if len(h_m) == 0:
            y2 = 0
        else:
            idxs = [colors.index( s.replace(' hair','') ) for s in h_m]
            y2 = random.choice(idxs)
        
        y1_buf.append(y1)
        y2_buf.append(y2)
        
        img_path = join('./faces',idx+'.jpg')
        x = imread(img_path)
        x = imresize(x, [64,64,3]).astype(np.uint8)
        x = x.reshape([1,64,64,3]) 
        x_buf.append(x)
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
