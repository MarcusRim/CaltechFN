## Instructions for running RetinaNet models

# Packages Needed:
1. pyyaml 5.1
2. torch (compiled with the correct version of cudatoolkits, follow https://pytorch.org/get-started/locally/)
3. torchvision
4. Detectron2 Installation (follow https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md).
    Note that to compile pycocotools, you will need to have VSCode C++ Developer Tools installed. 
5. pandas
6. h5py

# Run Instructions
1. Visit the datasets folder to ensure that all data (including the json files)  is properly available in the correct locations. 
2. For each python file, run the following command in terminal:
```shell
python retinanet_dataset1_on_dataset2.py
```
where dataset1 indicates the training dataset for the model, and dataset2 indicates the testing dataset for the model. We provide four combinations: svhn_on_caltechfn, svhn_on_svhn, caltechfn_on_svhn, caltechfn_on_caltechfn. 

To write the python code to train and test the RetinaNet models, we followed this tutorial: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=b2bjrfb2LDeo.
