import os
import gzip
import numpy as np
import pdb

def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def d_sigmoid(x):
    return x*(1 - x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def randomize_copies(a, b):
    assert len(a) == len(b)
    p1 = np.random.permutation(len(a))
    p2 = np.random.permutation(len(a))
    return a[p1], b[p2]

def d_RELU(x):
    return np.array(x>0).astype('double')

def RELU(x):
    return np.maximum(x, 0)

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a