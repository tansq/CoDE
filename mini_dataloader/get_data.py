import os
import numpy as np
import random

#./././ Read data from txt and shuffle them based seed ./././ #
def read_shuffle(txt, shuffle):
    file = open(txt, "r", encoding='utf-8')
    lines = file.read().split('\n')
    file.close()
    if shuffle:
        random.seed(2022)
        random.shuffle(lines)
        random.seed(None)
    return lines

def get_data(root, name, ratio, divide_shuffle=True, mode='train'):
    # ./././ read the corresponding datasets ./././ #
    root = './'
    if name.lower() == 'dataset':
        lines = read_shuffle(txt=os.path.join(root, 'dataset/data.txt'), shuffle=divide_shuffle)
        prefix = os.path.join(root, 'dataset')
    else:
        assert 'invalid datasets'

    # ./././ divide the train set and test set ./././ #
    assert mode == 'train' or mode == 'test' or mode == 'val', 'invalid mode!'
    if mode == 'train':
        flist = lines[:int(ratio * len(lines))]
    elif mode == 'test':
        flist = lines[int(ratio * len(lines)):]
    elif mode == 'val':
        flist = lines[int(ratio * len(lines)):]

    return prefix, flist