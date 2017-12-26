import os, sys
import argparse
import numpy as np
import random

from scipy.ndimage import imread
from scipy.misc import imresize, imsave
from os.path import join



labels = ['GreenStraightRight', 'off', 'GreenStraightLeft', 'GreenStraight', 'RedStraightLeft', 'GreenRight', 'Yellow', 'RedStraight', 'Green', 'GreenLeft', 'RedRight', 'RedLeft', 'Red']

label_dict = {label:i for i,label in enumerate(labels)}



def main():
    npz = np.load('./train.npz')
    x = npz['x_train']
    y = npz['y_train']
    for i in range(x.shape[0]):
        img = x[i,:,:,:]
        imsave('./out/%d.png' % i,img)
if __name__ == '__main__':
    main()
