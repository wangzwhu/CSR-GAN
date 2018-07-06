# Cascaded SR-GAN for Scale-Adaptive Low Resolution Person Re-identification

Demo code for Cascaded SR-GAN for Scale-Adaptive Low Resolution Person Re-identification in IJCAI-18. We use SALR-VIPeR as an example.


### Prepare the initial models and dataset.
> Download the [VGG model](https://pan.baidu.com/s/17164p0is8rc1G092dAmd6A) to current folder.
> Download the [Basic Re-id model](https://pan.baidu.com/s/1C4MtuUvo-jZdP1FIiIbXXQ), pre-trained on Market-1501, to current folder.
> Download the [dataset](https://pan.baidu.com/s/1OVOAR6Ga9qHCvi4RsgVXkA) to `.\dataset\`.

### Train CSR-GAN.
> Run `python train_csr_gan_viper.py`

### Extract Feature.
> Run `python get_feature_viper.py`

### Citation
Please cite this paper in your publications if it helps your research:
```
@inproceedings{wang2018cascaded,
  title={Cascaded SR-GAN for Scale-Adaptive Low Resolution Person Re-identification.},
  author={Wang, Zheng and Ye, Mang and Yang, Fan and Bai, Xiang and Satoh, Shin'ichi},
  booktitle={IJCAI},
  year={2018}
}
```

If you also use the [SALR datasets](https://pan.baidu.com/s/1boKBvrh), please kindly cite:

```
@inproceedings{wang2016scale,
  title={Scale-Adaptive Low-Resolution Person Re-Identification via Learning a Discriminating Surface.},
  author={Wang, Zheng and Hu, Ruimin and Yu, Yi and Jiang, Junjun and Liang, Chao and Wang, Jinqiao},
  booktitle={IJCAI},
  pages={2669--2675},
  year={2016}
}
```


Please also kindly cite:
```
@article{ledig2017photo,
  title={Photo-realistic single image super-resolution using a generative adversarial network},
  author={Ledig, Christian and Theis, Lucas and Husz{\'a}r, Ferenc and Caballero, Jose and Cunningham, Andrew and Acosta, Alejandro and Aitken, Andrew and Tejani, Alykhan and Totz, Johannes and Wang, Zehan and others},
  journal={arXiv preprint},
  year={2017}
}
```
This code helps us very much. Link: [https://github.com/tensorlayer/srgan](https://github.com/tensorlayer/srgan)


Contact: wangz@nii.ac.jp