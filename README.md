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
![batch_1_inputs](https://user-images.githubusercontent.com/16810812/43622189-e7730e68-96a8-11e8-9b00-e46d55bb8358.png)
![batch_1_outputs1](https://user-images.githubusercontent.com/16810812/43622190-e780db42-96a8-11e8-8144-8532ac112aa0.png)
![batch_15_inputs](https://user-images.githubusercontent.com/16810812/43622191-e790014e-96a8-11e8-8c5a-7947eff80b1f.png)
![batch_15_outputs1](https://user-images.githubusercontent.com/16810812/43622192-e79fedac-96a8-11e8-9dcc-7b0a73af0631.png)

## TO-DO

Run the training for Celebrity dataset to compare results with the paper
