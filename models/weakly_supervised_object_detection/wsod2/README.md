## Instructions for Running

We simply used the code from https://github.com/researchmm/WSOD2.

Please follow instructions on the relevant README file to prepare the dataset and run the model. You will have to adjust the files to run the models on the relevant datasets. Note, to obtain the .pkl files, you can follow the instructions from Wetectron.

The hyperparameters used for training on CaltechFN are:
```python
lr=0.005, momentum=0.9, weight_decay=0.001
```

The hyperparameters used for training on SVHN are:
```python
lr=0.001, momentum=0.9, weight_decay=0.001
```
