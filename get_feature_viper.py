import os, time, pickle, random, time
import numpy as np
import logging
import scipy.io as sio 
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import Model, load_model
from keras import backend as K 

import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import config, log_config


def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs


def evaluate():
    print("evaluate")
    tl.global_flag['mode'] = 'evaluate'
    save_dir_4_images = "./gen/"
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.img_path, regx='.*.bmp', printable=False))
    
    #config.gpu_options.allow_growth = True
    sess_id = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    set_session(sess_id)

    t_image_8 = tf.placeholder('float32', [None,  16,  6, 3], name='input_image_8')
    t_image_4 = tf.placeholder('float32', [None,  32,  12, 3], name='input_image_4')
    t_image_2 = tf.placeholder('float32', [None,  64,  24, 3], name='input_image_2')
    net_g_8 = SRGAN_g_8(t_image_8, is_train=False, reuse=False)
    net_g_4 = SRGAN_g_4(t_image_4, is_train=False, reuse=False)
    net_g_2 = SRGAN_g_2(t_image_2, is_train=False, reuse=False)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/viper_icsr_g_{}8.npz'.format('srgan'), network=net_g_8)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/viper_icsr_g_{}4.npz'.format('srgan'), network=net_g_4)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/viper_icsr_g_{}2.npz'.format('srgan'), network=net_g_2)

    net = load_model(checkpoint_dir+'/viper_icsr_re_id.npz')
    net = Model(input=net.input, output= [net.get_layer('avg_pool').output])

    start_time = time.time()
    valid_lr_imgs = read_all_imgs(valid_lr_img_list, path=config.VALID.img_path, n_threads=32)
    test_b_features = []

    for idx in range(0, len(valid_lr_imgs)):
        valid_lr_img = valid_lr_imgs[idx]
        size = valid_lr_img.shape
        print(size[0])   
        print(size[1])   
        scale = (float)(size[0])/128
        if scale <= 0.125: 
            lr_image = downsample_fn_16(valid_lr_img)  
            out8 = sess.run(net_g_8.outputs, {t_image_8: [lr_image]})
            out4 = sess.run(net_g_4.outputs, {t_image_4: [out8[0]]})
            out2 = sess.run(net_g_2.outputs, {t_image_2: [out4[0]]})
        elif 0.125< scale <= 0.25:
            lr_image = downsample_fn_32(valid_lr_img) 
            out4 = sess.run(net_g_4.outputs, {t_image_4: [lr_image]})
            out2 = sess.run(net_g_2.outputs, {t_image_2: [out4[0]]})			
        elif 0.25< scale <= 0.5:
            lr_image = downsample_fn_64(valid_lr_img)
            out2 = sess.run(net_g_2.outputs, {t_image_2: [lr_image]})	

        print("[*] save images")
        tl.vis.save_image(out2[0], save_dir_4_images+str(idx)+".bmp")

        img = upsample_fn_224(out2[0])
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)    
        feature = net.predict(img)
        test_b_features.append(np.squeeze(feature))

    sio.savemat('test_b_features.mat', {'test_b_features':test_b_features})
    print("took: %4.4fs" % (time.time() - start_time)) 

    valid_a_img_list = sorted(tl.files.load_file_list(path=config.VALID.img_path_2, regx='.*.bmp', printable=False))
    start_time = time.time()
    valid_a_imgs = read_all_imgs(valid_a_img_list, path=config.VALID.img_path_2, n_threads=32)
    test_a_features = []
    for idx in range(0, len(valid_a_imgs)):
        valid_lr_img = valid_a_imgs[idx]
        img = upsample_fn_224(valid_lr_img)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)    
        feature = net.predict(img)
        test_a_features.append(np.squeeze(feature))

    sio.savemat('test_a_features.mat', {'test_a_features':test_a_features})
    print("took: %4.4fs" % (time.time() - start_time)) 


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    evaluate()
