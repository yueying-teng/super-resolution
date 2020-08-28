## Single Image Super Resolution

### Purpose

This repo contains the code to carry out the following experimentations:

- EDSR model trained with L1 loss produces better results(measured by SSIM) than that trained with L2 loss.
- EDSR model finetuned with perceptual loss and MAE produces results that are more visually pleasing, though their SSIM is lower.

### Data

The pet dataset used in this repo can be downloaded from [here](https://www.robots.ox.ac.uk/~vgg/data/pets/).

### Results

EDSR model trained using L1 pixel loss.

![alt text](https://github.com/yueying-teng/super-resolution/blob/master/img_src/EDSR_MAE.png)

EDSR model trained using perceptual loss.

![alt text](https://github.com/yueying-teng/super-resolution/blob/master/img_src/EDSR_perceptual.png)

Comparison of the two EDSR models(L1 pixel and perceptual loss).

![alt text](https://github.com/yueying-teng/super-resolution/blob/master/img_src/comparison.png)
