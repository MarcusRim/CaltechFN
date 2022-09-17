## Instructions for Running

We simply used the code from https://github.com/shenyunhang/DRN-WSOD-pytorch/tree/DRN-WSOD/projects/WSL.

Please follow instructions on the relevant README file to prepare the dataset. 

The hyperparameters used for training on CaltechFN are:
```python
_C.SOLVER.BASE_LR = 0.005
_C.SOLVER.MOMENTUM = 0.93
_C.SOLVER.WEIGHT_DECAY = 0.001
```

The hyperparameters used for training on SVHN are:
```python
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.MOMENTUM = 0.93
_C.SOLVER.WEIGHT_DECAY = 0.005
```