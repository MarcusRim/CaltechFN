## Instructions for Running

We simply used the code from https://github.com/ppengtang/pcl.pytorch.

Please follow instructions on the relevant README file to prepare the dataset and run the model. 

The hyperparameters used for training on CaltechFN are:
```python
_C.SOLVER.BASE_LR = 0.0005
_C.SOLVER.MOMENTUM = 0.85
_C.SOLVER.WEIGHT_DECAY = 0.005
```

The hyperparameters used for training on SVHN are:
```python
_C.SOLVER.BASE_LR = 0.0005
_C.SOLVER.MOMENTUM = 0.85
_C.SOLVER.WEIGHT_DECAY = 0.005
```