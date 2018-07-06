import os, time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import config, log_config
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dense, Flatten, Dropout
from keras.initializers import RandomNormal
from keras.models import Model
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.applications.resnet50 import preprocess_input

num_classes = 528

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init    = config.TRAIN.lr_init
beta1      = config.TRAIN.beta1
## SRGAN
n_epoch_init = config.TRAIN.n_epoch_init
n_epoch      = config.TRAIN.n_epoch
lr_decay     = config.TRAIN.lr_decay
decay_every  = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))

def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    labels = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_labels    = range(idx, idx + n_threads, 1)
        b_labels    = [i/2 for i in b_labels]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        imgs.extend(b_imgs)
        labels.extend(b_labels)
        print('read %d from %s' % (len(imgs), path))
    return (imgs, labels)

def train():
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/srgan_ginit"
    save_dir_gan = "samples/srgan_gan"
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_img_1_list = sorted(tl.files.load_file_list(path=config.TRAIN.img_path_1, regx='.*.bmp', printable=False))
    (train_1_imgs, train_1_labels) = read_all_imgs(train_img_1_list, path=config.TRAIN.img_path_1, n_threads=24)
    train_1_labels = to_categorical(train_1_labels)

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image_8 = tf.placeholder('float32', [batch_size,  16,   6, 3],  name='t_image_8')
    t_image_4 = tf.placeholder('float32', [batch_size,  32,   12, 3], name='t_image_4')
    t_image_2 = tf.placeholder('float32', [batch_size,  64,   24, 3], name='t_image_2')
    t_image_1 = tf.placeholder('float32', [batch_size,  128,  48, 3], name='t_image_1')

    # G and D
    net_g_8                = SRGAN_g_8(t_image_8, is_train=True, reuse=False)
    net_g_4                = SRGAN_g_4(t_image_4, is_train=True, reuse=False)
    net_g_2                = SRGAN_g_2(t_image_2, is_train=True, reuse=False)
    net_d_8, logits_real_8 = SRGAN_d_8(t_image_4, is_train=True, reuse=False)
    net_d_4, logits_real_4 = SRGAN_d_4(t_image_2, is_train=True, reuse=False)
    net_d_2, logits_real_2 = SRGAN_d_2(t_image_1, is_train=True, reuse=False)
    _,       logits_fake_8 = SRGAN_d_8(net_g_8.outputs,  is_train=True, reuse=True)
    _,       logits_fake_4 = SRGAN_d_4(net_g_4.outputs,  is_train=True, reuse=True)
    _,       logits_fake_2 = SRGAN_d_2(net_g_2.outputs,  is_train=True, reuse=True)

    net_g_8.print_params(False)
    net_g_4.print_params(False)
    net_g_2.print_params(False)
    net_d_8.print_params(False)
    net_d_4.print_params(False)
    net_d_2.print_params(False)

    # VGG
    t_image_1_224        = tf.image.resize_images(t_image_1, size=[224, 224], method=0, align_corners=False)
    t_image_2_224        = tf.image.resize_images(t_image_2, size=[224, 224], method=0, align_corners=False) 
    t_image_4_224        = tf.image.resize_images(t_image_4, size=[224, 224], method=0, align_corners=False)
    t_net_g_2_output_224 = tf.image.resize_images(net_g_2.outputs, size=[224, 224], method=0, align_corners=False)
    t_net_g_4_output_224 = tf.image.resize_images(net_g_4.outputs, size=[224, 224], method=0, align_corners=False)
    t_net_g_8_output_224 = tf.image.resize_images(net_g_8.outputs, size=[224, 224], method=0, align_corners=False)
	
    net_image_1_vgg,    vgg_image_1_emb         = Vgg19_simple_api((t_image_1_224+1)/2,  reuse=False)
    net_image_2_vgg,    vgg_image_2_emb         = Vgg19_simple_api((t_image_2_224+1)/2,  reuse=True)
    net_image_4_vgg,    vgg_image_4_emb         = Vgg19_simple_api((t_image_4_224+1)/2,  reuse=True)
    net_g_2_output_vgg, vgg_net_g_2_output_emb  = Vgg19_simple_api((t_net_g_2_output_224+1)/2, reuse=True) 
    net_g_4_output_vgg, vgg_net_g_4_output_emb  = Vgg19_simple_api((t_net_g_4_output_224+1)/2, reuse=True)
    net_g_8_output_vgg, vgg_net_g_8_output_emb  = Vgg19_simple_api((t_net_g_8_output_224+1)/2, reuse=True)

	# re-id
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))
    x = base_model.output
    x = Flatten()(x)
    x = Dropout(0.9)(x)
    id_preds = Dense(num_classes, activation='softmax', name='id_preds', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(x)
    re_id_net = Model(input=[base_model.input], output=[id_preds])
    for layer in re_id_net.layers:
        layer.trainable = True
    re_id_net.load_weights('basic_train_on_market.ckpt', by_name=True)
    

    ## test inference
    net_g_test_8 = SRGAN_g_8(t_image_8, is_train=False, reuse=True)
    net_g_test_4 = SRGAN_g_4(t_image_4, is_train=False, reuse=True)
    net_g_test_2 = SRGAN_g_2(t_image_2, is_train=False, reuse=True)
 
	
    # ###========================== DEFINE TRAIN OPS ==========================###
    # d loss
    d_loss1_8 = tl.cost.sigmoid_cross_entropy(logits_real_8, tf.ones_like(logits_real_8),  name='d1_8')
    d_loss1_4 = tl.cost.sigmoid_cross_entropy(logits_real_4, tf.ones_like(logits_real_4),  name='d1_4')
    d_loss1_2 = tl.cost.sigmoid_cross_entropy(logits_real_2, tf.ones_like(logits_real_2),  name='d1_2')
    d_loss2_8 = tl.cost.sigmoid_cross_entropy(logits_fake_8, tf.zeros_like(logits_fake_8), name='d2_8')
    d_loss2_4 = tl.cost.sigmoid_cross_entropy(logits_fake_4, tf.zeros_like(logits_fake_4), name='d2_4')
    d_loss2_2 = tl.cost.sigmoid_cross_entropy(logits_fake_2, tf.zeros_like(logits_fake_2), name='d2_2')
    d_loss_8 = d_loss1_8 + d_loss2_8
    d_loss_4 = d_loss1_4 + d_loss2_4
    d_loss_2 = d_loss1_2 + d_loss2_2

    # g loss
    g_gan_loss_8 = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake_8, tf.ones_like(logits_fake_8), name='g_8')
    g_gan_loss_4 = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake_4, tf.ones_like(logits_fake_4), name='g_4')
    g_gan_loss_2 = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake_2, tf.ones_like(logits_fake_2), name='g_2')
    mse_loss_8   = tl.cost.mean_squared_error(net_g_8.outputs, t_image_4, is_mean=True)
    mse_loss_4   = tl.cost.mean_squared_error(net_g_4.outputs, t_image_2, is_mean=True)
    mse_loss_2   = tl.cost.mean_squared_error(net_g_2.outputs, t_image_1, is_mean=True)
    vgg_loss_2   = 2e-6 * tl.cost.mean_squared_error(vgg_net_g_2_output_emb.outputs, vgg_image_1_emb.outputs, is_mean=True) # 2e-6
    vgg_loss_4   = 2e-6 * tl.cost.mean_squared_error(vgg_net_g_4_output_emb.outputs, vgg_image_2_emb.outputs, is_mean=True) # 2e-6
    vgg_loss_8   = 2e-6 * tl.cost.mean_squared_error(vgg_net_g_8_output_emb.outputs, vgg_image_4_emb.outputs, is_mean=True) # 2e-6
    vgg_id_loss   = tl.cost.mean_squared_error(net_g_2_output_vgg.outputs, net_image_1_vgg.outputs, is_mean=True)
	
    g_loss_8 = mse_loss_8 + g_gan_loss_8 + vgg_loss_8
    g_loss_4 = mse_loss_4 + g_gan_loss_4 + vgg_loss_4
    g_loss_2 = mse_loss_2 + g_gan_loss_2 + vgg_loss_2 + vgg_id_loss

    g_vars_8 = tl.layers.get_variables_with_name('SRGAN_g_8', True, True)
    g_vars_4 = tl.layers.get_variables_with_name('SRGAN_g_4', True, True)
    g_vars_2 = tl.layers.get_variables_with_name('SRGAN_g_2', True, True)
    d_vars_8 = tl.layers.get_variables_with_name('SRGAN_d_8', True, True)
    d_vars_4 = tl.layers.get_variables_with_name('SRGAN_d_4', True, True)
    d_vars_2 = tl.layers.get_variables_with_name('SRGAN_d_2', True, True)
	
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)

    ## Pretrain
    g_optim_init_8 = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss_8, var_list=g_vars_8)
    g_optim_init_4 = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss_4, var_list=g_vars_4)
    g_optim_init_2 = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss_2, var_list=g_vars_2)

    ## SRGAN
    g_optim_8 = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss_8, var_list=g_vars_8)
    g_optim_4 = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss_4, var_list=g_vars_4)
    g_optim_2 = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss_2, var_list=g_vars_2)
    d_optim_8 = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss_8, var_list=d_vars_8)
    d_optim_4 = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss_4, var_list=d_vars_4)
    d_optim_2 = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss_2, var_list=d_vars_2)

    re_id_net.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss={'id_preds': 'categorical_crossentropy'})

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/viper_icsr_g_srgan8.npz', network=net_g_8) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/viper_icsr_g_srgan_init8.npz', network=net_g_8)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/viper_icsr_d_srgan8.npz', network=net_d_8)
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/viper_icsr_g_srgan4.npz', network=net_g_4) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/viper_icsr_g_srgan_init4.npz', network=net_g_4)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/viper_icsr_d_srgan4.npz', network=net_d_4)
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/viper_icsr_g_srgan2.npz', network=net_g_2) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/viper_icsr_g_srgan_init2.npz', network=net_g_2)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/viper_icsr_d_srgan2.npz', network=net_d_2)

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()
    params = []
    for val in sorted( npz.items() ):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_image_1_vgg)

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    sample_imgs     = train_1_imgs[0:batch_size]
    sample_imgs_128 = tl.prepro.threading_data(sample_imgs, fn=downsample_fn_128)
    sample_imgs_64  = tl.prepro.threading_data(sample_imgs_128, fn=downsample_fn_64)
    sample_imgs_32  = tl.prepro.threading_data(sample_imgs_64, fn=downsample_fn_32)
    sample_imgs_16  = tl.prepro.threading_data(sample_imgs_32, fn=downsample_fn_16)
    print('sample 128 sub-image:', sample_imgs_128.shape, sample_imgs_128.min(), sample_imgs_128.max())
    print('sample 64 sub-image:', sample_imgs_64.shape,  sample_imgs_64.min(), sample_imgs_64.max())
    print('sample 32 sub-image:', sample_imgs_32.shape,  sample_imgs_32.min(), sample_imgs_32.max())
    print('sample 16 sub-image:', sample_imgs_16.shape,  sample_imgs_16.min(), sample_imgs_16.max())
    tl.vis.save_images(sample_imgs_16,  [ni, ni], save_dir_ginit+'/_train_sample_16.png')
    tl.vis.save_images(sample_imgs_32,  [ni, ni], save_dir_ginit+'/_train_sample_32.png')
    tl.vis.save_images(sample_imgs_64,  [ni, ni], save_dir_ginit+'/_train_sample_64.png')
    tl.vis.save_images(sample_imgs_128, [ni, ni], save_dir_ginit+'/_train_sample_128.png')
    tl.vis.save_images(sample_imgs_16,  [ni, ni], save_dir_gan+'/_train_sample_16.png')
    tl.vis.save_images(sample_imgs_32,  [ni, ni], save_dir_gan+'/_train_sample_32.png')
    tl.vis.save_images(sample_imgs_64,  [ni, ni], save_dir_gan+'/_train_sample_64.png')
    tl.vis.save_images(sample_imgs_128, [ni, ni], save_dir_gan+'/_train_sample_128.png')
	
    ###========================= initialize G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    for epoch in range(0, n_epoch_init+1):
        epoch_time = time.time()
        total_mse_loss_8, total_mse_loss_4, total_mse_loss_2, n_iter = 0, 0, 0, 0
        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_1_imgs), batch_size):
            step_time = time.time()
            b_imgs_128 = tl.prepro.threading_data(
                    train_1_imgs[idx : idx + batch_size],
                    fn=downsample_fn_128)
            b_imgs_64 = tl.prepro.threading_data(b_imgs_128, fn=downsample_fn_64)
            b_imgs_32 = tl.prepro.threading_data(b_imgs_64, fn=downsample_fn_32)
            b_imgs_16 = tl.prepro.threading_data(b_imgs_32, fn=downsample_fn_16)
            ## update G
            errM8, _ = sess.run([mse_loss_8, g_optim_init_8], {t_image_8: b_imgs_16, t_image_4: b_imgs_32})
            errM4, _ = sess.run([mse_loss_4, g_optim_init_4], {t_image_4: b_imgs_32, t_image_2: b_imgs_64})
            errM2, _ = sess.run([mse_loss_2, g_optim_init_2], {t_image_2: b_imgs_64, t_image_1: b_imgs_128})

			# re-id
            batch_label = train_1_labels[idx : idx + batch_size]
            b_imgs_224 = tl.prepro.threading_data(b_imgs_128, fn=upsample_fn_224)
            b_imgs_224 = preprocess_input(b_imgs_224)
            b_imgs_224 = np.array(b_imgs_224)
            hist = re_id_net.fit([b_imgs_224], {'id_preds': batch_label}, verbose = 0) 

            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f, %.8f, %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM8, errM4, errM2))
            total_mse_loss_8 += errM8
            total_mse_loss_4 += errM4
            total_mse_loss_2 += errM2
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f, %.8f, %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss_8/n_iter, total_mse_loss_4/n_iter, total_mse_loss_2/n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            out8 = sess.run(net_g_test_8.outputs, {t_image_8: b_imgs_16})#; print('gen sub-image:', out.shape, out.min(), out.max())
            out4 = sess.run(net_g_test_4.outputs, {t_image_4: b_imgs_32})#; print('gen sub-image:', out.shape, out.min(), out.max())
            out2 = sess.run(net_g_test_2.outputs, {t_image_2: b_imgs_64})#; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out8, [ni, ni], save_dir_ginit+'/train8_%d.png' % epoch)
            tl.vis.save_images(out4, [ni, ni], save_dir_ginit+'/train4_%d.png' % epoch)
            tl.vis.save_images(out2, [ni, ni], save_dir_ginit+'/train2_%d.png' % epoch)
			
        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_g_8.all_params, name=checkpoint_dir+'/viper_icsr_g_srgan_init8.npz', sess=sess)
            tl.files.save_npz(net_g_4.all_params, name=checkpoint_dir+'/viper_icsr_g_srgan_init4.npz', sess=sess)
            tl.files.save_npz(net_g_2.all_params, name=checkpoint_dir+'/viper_icsr_g_srgan_init2.npz', sess=sess)
            re_id_net.save(checkpoint_dir+'/viper_icsr_re_id.npz')

    ###========================= train GAN (SRGAN) =========================###
    for epoch in range(0, n_epoch+1):
        ## update learning rate
        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss_8, total_d_loss_4, total_d_loss_2, total_g_loss_8, total_g_loss_4, total_g_loss_2, n_iter = 0, 0, 0, 0, 0, 0, 0

        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_1_imgs), batch_size):
            step_time = time.time()
            b_imgs_128 = tl.prepro.threading_data(
                    train_1_imgs[idx : idx + batch_size],
                    fn=downsample_fn_128)
            b_imgs_64 = tl.prepro.threading_data(b_imgs_128, fn=downsample_fn_64)
            b_imgs_32 = tl.prepro.threading_data(b_imgs_64, fn=downsample_fn_32)
            b_imgs_16 = tl.prepro.threading_data(b_imgs_32, fn=downsample_fn_16)
            ## update D
            errD8, _ = sess.run([d_loss_8, d_optim_8], {t_image_8: b_imgs_16, t_image_4: b_imgs_32})
            errD4, _ = sess.run([d_loss_4, d_optim_4], {t_image_4: b_imgs_32, t_image_2: b_imgs_64})
            errD2, _ = sess.run([d_loss_2, d_optim_2], {t_image_2: b_imgs_64, t_image_1: b_imgs_128})

            ## update G
            errG8, errM8, errV8,         errA8, _ = sess.run([g_loss_8, mse_loss_8, vgg_loss_8,              g_gan_loss_8, g_optim_8], {t_image_8: b_imgs_16, t_image_4: b_imgs_32})
            errG4, errM4, errV4,         errA4, _ = sess.run([g_loss_4, mse_loss_4, vgg_loss_4,              g_gan_loss_4, g_optim_4], {t_image_4: b_imgs_32, t_image_2: b_imgs_64})
            errG2, errM2, errV2, errVID, errA2, _ = sess.run([g_loss_2, mse_loss_2, vgg_loss_2, vgg_id_loss, g_gan_loss_2, g_optim_2], {t_image_2: b_imgs_64, t_image_1: b_imgs_128})
			
			# re-id
            batch_label = train_1_labels[idx : idx + batch_size]
            b_imgs_224 = tl.prepro.threading_data(b_imgs_128, fn=upsample_fn_224)
            b_imgs_224 = preprocess_input(b_imgs_224)
            b_imgs_224 = np.array(b_imgs_224)
            hist = re_id_net.fit([b_imgs_224], {'id_preds': batch_label}, verbose = 0) 

            out8 = sess.run(net_g_test_8.outputs, {t_image_8: b_imgs_16}) 
            out4 = sess.run(net_g_test_4.outputs, {t_image_4: out8}) 
            out2 = sess.run(net_g_test_2.outputs, {t_image_2: out4}) 
            b_imgs_224 = tl.prepro.threading_data(out2, fn=upsample_fn_224)
            b_imgs_224 = preprocess_input(b_imgs_224)
            b_imgs_224 = np.array(b_imgs_224)			
            hist = re_id_net.fit([b_imgs_224], {'id_preds': batch_label}, verbose = 0)			

            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" % (epoch, n_epoch, n_iter, time.time() - step_time,             errD8, errG8, errM8, errV8, errA8))
            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" % (epoch, n_epoch, n_iter, time.time() - step_time,             errD4, errG4, errM4, errV4, errA4))
            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f vggid: %.6f)" % (epoch, n_epoch, n_iter, time.time() - step_time, errD2, errG2, errM2, errV2, errA2, errVID))
            total_d_loss_8 += errD8
            total_d_loss_4 += errD4
            total_d_loss_2 += errD2
            total_g_loss_8 += errG8
            total_g_loss_4 += errG4
            total_g_loss_2 += errG2
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss_8/n_iter, total_g_loss_8/n_iter)
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss_4/n_iter, total_g_loss_4/n_iter)
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss_2/n_iter, total_g_loss_2/n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            out8 = sess.run(net_g_test_8.outputs, {t_image_8: b_imgs_16})#; print('gen sub-image:', out.shape, out.min(), out.max())
            out4 = sess.run(net_g_test_4.outputs, {t_image_4: b_imgs_32})#; print('gen sub-image:', out.shape, out.min(), out.max())
            out2 = sess.run(net_g_test_2.outputs, {t_image_2: b_imgs_64})#; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out8, [ni, ni], save_dir_gan+'/train8_%d.png' % epoch)
            tl.vis.save_images(out4, [ni, ni], save_dir_gan+'/train4_%d.png' % epoch)
            tl.vis.save_images(out2, [ni, ni], save_dir_gan+'/train2_%d.png' % epoch)
			
        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_g_8.all_params, name=checkpoint_dir+'/viper_icsr_g_srgan8.npz', sess=sess)
            tl.files.save_npz(net_g_4.all_params, name=checkpoint_dir+'/viper_icsr_g_srgan4.npz', sess=sess)
            tl.files.save_npz(net_g_2.all_params, name=checkpoint_dir+'/viper_icsr_g_srgan2.npz', sess=sess)
            tl.files.save_npz(net_d_8.all_params, name=checkpoint_dir+'/viper_icsr_d_srgan8.npz', sess=sess)
            tl.files.save_npz(net_d_4.all_params, name=checkpoint_dir+'/viper_icsr_d_srgan4.npz', sess=sess)
            tl.files.save_npz(net_d_2.all_params, name=checkpoint_dir+'/viper_icsr_d_srgan2.npz', sess=sess)
            re_id_net.save(checkpoint_dir+'/viper_icsr_re_id.npz')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    train()

