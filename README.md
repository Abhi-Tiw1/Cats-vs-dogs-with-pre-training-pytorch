# Cats vs Dogs Classification using self defined & pre-trained networks

This repository has been adapted from the pytorch template repository that can be found [here](https://github.com/victoresque/pytorch-template).
The purpose of this repository is to test pre-training benifits and transfer learning on the popular cats vs dogs dataset from kaggle.

## Additions to the repository

- A Dataset and DataLoader class was added to load the image files
- Config file can now to use to directly select a pretrained model
- First all layers of the model are frozen by setting
> variable.requires_grad = False with all variables iterated using model.parameters()
- Next the models fully connected layer is changed to create my own layers and final softmax function for binary classification
- Both *ResNet18* and *ResNet50* models can now be used for this task

## Performace using the different models

