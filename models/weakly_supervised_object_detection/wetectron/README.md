## Instructions for Running

We simply used the code from https://github.com/NVlabs/wetectron.

Please follow instructions on the relevant README file and on https://github.com/NVlabs/wetectron/blob/master/docs/USE_YOUR_OWN_DATA.md to prepare the dataset. 

The hyperparameters used for training on CaltechFN are:
```python
_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.MOMENTUM = 0.915
_C.SOLVER.WEIGHT_DECAY = 0.0001
```

The hyperparameters used for training on SVHN are:
```python
_C.SOLVER.BASE_LR = 0.005
_C.SOLVER.MOMENTUM = 0.945
_C.SOLVER.WEIGHT_DECAY = 0.0001
```