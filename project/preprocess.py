import os, sys
import argparse
import numpy as np
import random

from scipy.ndimage import imread
from scipy.misc import imresize, imsave
from os.path import join

#default_height = 50
#default_width = 25
default_size = (50, 50)


labels = ['GreenStraightRight', 'off', 'GreenStraightLeft', 'GreenStraight', 'RedStraightLeft', 'GreenRight', 'Yellow', 'RedStraight', 'Green', 'GreenLeft', 'RedRight', 'RedLeft', 'Red']

label_dict = {label:i for i,label in enumerate(labels)}

def parse():
    parser = argparse.ArgumentParser(description='utils')
    parser.add_argument('input', help='png image files directory')
    parser.add_argument('valid_dir', help='png image files directory')
    parser.add_argument('test_dir', help='png image files directory')
    parser.add_argument('-o','--output', default='./output.npz', help='path to save npz file.')
    parser.add_argument('-q','--quiet', action='store_true', default=True, help='show the log')
    parser.add_argument('--npz', action='store_true', default=False, help='compress data to npz file, include preprocessing.')
    parser.add_argument('--split_valid', type=float, default=None, help='split training data into (train, valid) dataset with given ratio. 0.1 -> (valid/total)=0.1')
    args = parser.parse_args()
    return args

def read_dir(dir_path):
    file_list = [join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.png')] 
    print('read directory %s with %d files.' % (dir_path, len(file_list)) )
    #data
    data_buf = []
    for f in file_list:
        img = imread(f)
        assert len(img.shape) == 3
        img = normalize_img(img)
        h,w,dep = img.shape
        img = img.reshape([1,h,w,dep])
        data_buf.append(img)
    
    #label
    label_buf = []
    for f in file_list:
        name = os.path.basename(f)
        label = name.split('_')[-2]
        #RED
        label = label_dict[label]
        label_buf.append( np.array([label]).reshape([1,1]) )


    return np.vstack(data_buf), np.vstack(label_buf)
def ceil(x):
    return int(x)+1 if float(x) > int(x) else int(x)
def floor(x):
    return int(x)
def pad2square(rect):
    # h, w, dep

    assert len(rect.shape) == 3
    h, w, dep = rect.shape
    size = max(h,w)
    dis = abs(h-w)
    pad_a = floor( float(dis)/2 )
    pad_b = ceil( float(dis)/2 )
    
    if h >= w:#pad width
        pad = ((0,0), (pad_a, pad_b), (0,0))
    else:
        pad = ((pad_a, pad_b), (0,0), (0,0))

    square = np.lib.pad(rect, pad, 'constant', constant_values=0)
    return square
def normalize_img(img):
    
    square = pad2square(img)
    img = imresize(square, default_size) 
    return img
def split_valid(input_dir, output_dir, ratio):
    from shutil import copyfile
    file_list = [join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.png')]
    random.shuffle(file_list)
    
    split = int( len(file_list) * ratio )
    valid, train = file_list[:split], file_list[split:]

    os.mkdir(output_dir)
    os.mkdir(join(output_dir,'train'))
    os.mkdir(join(output_dir,'valid'))
    #data
    for f in train:
        dst = join(output_dir, 'train', os.path.basename(f))
        copyfile(f, dst)
    for f in valid:
        dst = join(output_dir, 'valid', os.path.basename(f))
        copyfile(f, dst)


def main():
    args = parse()
    if args.npz:

        x_train, y_train = read_dir(args.input)
        x_valid, y_valid = read_dir(args.valid_dir)
        x_test, y_test  = read_dir(args.test_dir)
        print('start writing output file %s' % args.output)
        with open(args.output,'wb') as output:
            np.savez(output, x_train=x_train, y_train=y_train,
                    x_valid=x_valid, y_valid=y_valid,
                    x_test=x_test, y_test=y_test)
        '''
        with open(args.output,'wb') as output:
            np.savez(output, x_train=x_train, y_train=y_train)
        print('Done')
        '''
    elif args.split_valid is not None:
        split_valid(args.input, args.output, args.split_valid)


if __name__ == '__main__':
    main()
