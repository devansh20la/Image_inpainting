# Image In-painting with irregular holes
[Un-Official] implementation of the paper "Image In-painting with irregular holes" by Nvidia [https://arxiv.org/abs/1804.07723]

This project was a part of my bigger project regarding medical image analysis

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

Here is a list of some very specific dependencies
```
Pytorch
visdom (I use it to log loss values during training)
```
## Data Loader

You should place all the files inside the data folder arranged with following naming format.

```

Input Images:  XXXXX_input.png
Mask: 1) XXXXXX_segmentation.png
      2) XXXXXX_mask.png
Target: XXXXXX_target.png

```
### Data-augmentation
1) The data loader randomly generates a mask to augment the data during training.
2) I used segmentation image to mask the input image to help model learn the features from the skin region.
3) Furthermore, we used flipping and random crops to further augment the data during training

## Training

```
CUDA_VISIBLE_DEVICES=1,2,3 python Train_Gen.py --bs=25 --ms=123 --glr=2e-4  --ep=1000 --n=hair_removal
```

## Results
