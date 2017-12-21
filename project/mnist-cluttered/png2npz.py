import os, sys
import argparse
import numpy as np

from scipy.ndimage import imread
from os.path import join

def parse():
    parser = argparse.ArgumentParser(description='utils')
    parser.add_argument('train_dir', help='png image files directory')
    parser.add_argument('valid_dir', help='png image files directory')
    parser.add_argument('test_dir', help='png image files directory')
    parser.add_argument('-p','--preprocess', action='store_true', default=False, help='preprocess the json file')
    parser.add_argument('-m','--merge', action='store_true', default=False, help='merge multiple json file')
    parser.add_argument('-o','--output', default='./output.npz', help='path to save npz file.')
    parser.add_argument('-q','--quiet', action='store_true', default=True, help='show the log')
    args = parser.parse_args()
    return args

def read_dir(dir_path):
    file_list = [join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.png')] 

    #get sample info : width, height, channels
    sample = file_list[0]
    img = imread(sample)
    
    #(60, 60)
    assert len(img.shape) == 2
    
    w, h = img.shape
    num = len(file_list)
    
    #data
    data_buf = []
    for f in file_list:
        img = imread(sample)
        assert img.shape == (w, h)
        img = img.reshape([1, w, h])
        data_buf.append(img)
    
    #label
    label_buf = []
    for f in file_list:
        name = os.path.basename(f)
        buf = name.split('_')[-1]
        #3.png
        label = int(buf[:-4])
        label_buf.append( np.array([label]).reshape([1,1]) )


    return np.vstack(data_buf), np.vstack(label_buf)
    


def main():
    args = parse()
    x_train, y_train = read_dir(args.train_dir)
    x_valid, y_valid = read_dir(args.valid_dir)
    x_test, y_test  = read_dir(args.test_dir)
    print x_train.shape, y_train.shape
    print('start writing output file %s' % args.output)
    with open(args.output,'w') as output:
        np.savez(output, x_train=x_train, y_train=y_train,
                x_valid=x_valid, y_valid=y_valid,
                x_test=x_test, y_test=y_test)
    print('Done')    

if __name__ == '__main__':
    main()
