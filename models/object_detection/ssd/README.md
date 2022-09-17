## Instructions for Running

We simply used the code from https://github.com/lufficc/SSD.

To train the model, first set up directories as the instructions say. Since we provide annotations in COCO format, please follow the COCO setup format. Please follow the DEVELOP_GUIDE.md to set up the datasets.

The hyperparameters used for training on CaltechFN are: 
```python
SOLVER:
    LR: 0.001 
    MOMENTUM: 0.9
    WEIGHT_DECAY: 0.005
```

The hyperparameters used for training on SVHN are: 
```python
SOLVER:
    LR: 0.0005 
    MOMENTUM: 0.93
    WEIGHT_DECAY: 0.0001
```