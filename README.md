## Single Image Super Resolution

### Purpose

This repo contains the code to carry out the following experimentations:

- EDSR model trained with L1 loss produces better results(measured by SSIM) than that trained with L2 loss.
- EDSR model finetuned with a mixture of perceptual loss and MAE loss produces results with lower SSIMs but are more visually pleasing.

### Data

The pet dataset used in this repo can be downloaded from [here](https://www.robots.ox.ac.uk/~vgg/data/pets/).

### Results

EDSR model trained using L1 pixel loss.

![alt text](https://github.com/yueying-teng/super-resolution/blob/master/img_src/EDSR_MAE.png)

EDSR model finetuned using L1 pixel and perceptual loss.

![alt text](https://github.com/yueying-teng/super-resolution/blob/master/img_src/EDSR_perceptual.png)

Comparison of the two EDSR models(L1 pixel and perceptual loss) above.
Note that though EDSR model trained using L1 pixel gives higher SSIMs, it does not produce images with as much real texture. e.g. cat whiskers, dog fur.

![alt text](https://github.com/yueying-teng/super-resolution/blob/master/img_src/comparison.png)

### Usage

Build the container.

```bash
$ sudo make init-train
```

After the the container is built. Run and ssh into the container.

```bash
$ sudo make run-container
```

Once in the container. Start jupyter notebook.

```bash
$ make jupyter
```

To reproduce the EDSR model finetuned with perceptual loss, use the following code in a notebook.

```python
import sys
import os
sys.path.append("../")

from EDSR import main as main
from EDSR import config as config
# fintune the EDSR model trained using L1 pixel loss using both perceptual loss and L1 pixel loss
main.train()

# show prediction results from the finetuned model
weight_path1 = config.PRETRAINED_WEIGHT
weight_path2 = os.path.join('../model', '{}.h5'.format(config.FINETUNED_MODEL_NAME))

main.show_one_model_results(weight_path=weight_path2)
# compare the prediction results from two EDSR models, one trained on L1 pixel loss, the other trained on perceptual loss and L1 pixel loss
main.compare_two_models_results(weight_path1=weight_path1, weight_path2=weight_path2, keep_dim=False, 
                                show_ori_img=True, multi_255=True, show_patch=True)

```
