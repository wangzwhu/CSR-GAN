from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()
config.VALID = edict()

## Adam
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## SRGAN
config.TRAIN.n_epoch_init = 100
config.TRAIN.n_epoch = 2000
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

config.TRAIN.img_path_1 = './dataset/SALR-VIPeR/cam_training/'
config.TRAIN.img_path_2 = './dataset/SALR-VIPeR/cam_training/'
config.VALID.img_path   = './dataset/SALR-VIPeR/cam_test_b/'
config.VALID.img_path_2 = './dataset/SALR-VIPeR/cam_test_a/'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
