#!/usr/bin/env python
"""
Quick sample script that crops the traffic light images with label class.

Example usage:
    python crop.py input.yaml output_folder
"""
import sys
import os
import cv2
import numpy as np
import argparse
from read_label_file import get_all_labels

def parse():
    parser = argparse.ArgumentParser(description='crop utils')
    parser.add_argument('input_dir', help='png image files directory')
    parser.add_argument('-o','--output', default='./crop_data', help='path to save cropped png files.')
    args = parser.parse_args()
    return args

labels = ['GreenStraightRight', 'off', 'GreenStraightLeft', 'GreenStraight', 'RedStraightLeft', 'GreenRight', 'Yellow', 'RedStraight', 'Green', 'GreenLeft', 'RedRight', 'RedLeft', 'Red']

label_dict = {label:i for i,label in enumerate(labels)}


def ir(some_value):
    """Int-round function for short array indexing """
    return int(round(some_value))

def crop_label_images(input_yaml, output_folder=None):
    """
    Shows and draws pictures with labeled traffic lights.
    Can save pictures.

    :param input_yaml: Path to yaml file
    :param output_folder: If None, do not save picture. Else enter path to folder
    """
    images = get_all_labels(input_yaml)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, image_dict in enumerate(images):
        image = cv2.imread(image_dict['path'])
        if image is None:
            raise IOError('Could not open image path', image_dict['path'])

        for idx,box in enumerate(image_dict['boxes']):
            '''
            print type(image)
            print image.shape
            cv2.rectangle(image,
                          (ir(box['x_min']), ir(box['y_min'])),
                          (ir(box['x_max']), ir(box['y_max'])),
                          (0, 255, 0))
            print box
            print image_dict['path']
            sys.exit()
            '''
            y1,y2,x1,x2 = ir(box['y_min']), ir(box['y_max']), ir(box['x_min']),ir(box['x_max'])
            y1 = min(max(0,y1),720)
            y2 = min(max(0,y2),720)
            x1 = min(max(0,x1),1280)
            x2 = min(max(0,x2),1280)
            if y2 > y1 and  x2 > x1:
                img = image[y1:y2, x1:x2, :]
                cv2.imwrite(os.path.join(output_folder, str(i).zfill(10) + '_'
                            + str(idx) + '_' + box['label'] + '_'
                            + os.path.basename(image_dict['path'])), img)


def main():
    args = parse()
    
    crop_label_images(args.input_dir, args.output)

if __name__ == '__main__':
    main()
