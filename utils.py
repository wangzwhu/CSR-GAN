import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=384, hrg=384, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def crop_sub_imgs_fn_128(x, is_random=True):
    x = crop(x, wrg=128, hrg=128, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def upsample_fn_224(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[224, 224], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x
	
def downsample_fn_128(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[128, 48], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_fn_64(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[64, 24], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_fn_32(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[32, 12], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_fn_16(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[16, 6], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x


