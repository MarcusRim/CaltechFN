## Instructions for running ResNet models

# Packages Needed:
1. tensorflow 2.2+
2. PIL 
3. scipy

# Run Instructions
1. Visit the datasets folder to ensure that all data is properly available in the correct locations. 
2. For each python file, run the following command in terminal:
```shell
python resnet_dataset1_on_dataset2.py
```
where dataset1 indicates the training dataset for the model, and dataset2 indicates the testing dataset for the model. We provide four combinations: svhn_on_caltechfn, svhn_on_svhn, caltechfn_on_svhn, caltechfn_on_caltechfn. 
