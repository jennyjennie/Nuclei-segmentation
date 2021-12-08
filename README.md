# Nuclei-segmentation

## Description

Instance detection on nucleus dataset using mmdetection for VRDL HW3

## Requirements

In this project, mmdetection was used. Please refer to [mmdetection](https://github.com/open-mmlab/mmdetection.git) for installation.

To install other requirements:

```setup
pip install -r requirements.txt
```
## Dataset Preparation
#### Prepare annotations
+ Use ```datarpcess/mask_imgs_to_json.py```  to generate ```train.json```and ```test.json```

#### Project structure
```
mmdetection 
└─── coco
│    │
│    └─── annotations
│    │    │  train.json
│    │    |  test.json
│    │
│    └─── images
│         │  1.png
│         |  2.png
│         |  ...
│
└─── config 
│    └─── override_configs
│    │    │  ${MY_CONFIGS_FILE}.py
│    │    |  ...
└─── ...

```
> ```{MY_CONFIGS_FILE}.py``` used in this project can be found in ```src``` file in this repository.

## Training

To train the model used in this project, run this command:

```train
cd mmdetection
python tools/train.py ./configs/override_configs/${MY_CONFIGS_FILE}.py
```


## Test

To test the trained model, run:

```test
python tools/test.py ./configs/override_configs/${MY_CONFIGS_FILE}.py --format-only --options jsonfile_prefix=${JSONFILE_PREFIX}
```

## Pre-trained Models

The pretrained model can be downloaded in [mmdetection model zoo](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn):

The trained model for this project can be downloaded here:

https://drive.google.com/drive/folders/1_KiaGKaL7J4J23uoZYN6PWtLbJQAHxN2?usp=sharing

## Results

The model achieves the following performance on :


| Model name         | mAP  |
| ------------------ |---------------- |
| Mask-rcnn   |     0.23336     |    

