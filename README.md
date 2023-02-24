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

## Performance using the different models
Model HP: Loss = cross entropy/no regularization, Optimzer = Adam, Batch size = 128

- Model 1: CatsvsDogs ConvNet (trained from scratch) 
 - Number of epochs = 5, 
 - **Train Accuracy: 88.93%**
 - **Validation Accuracy: 86.56%**

- Model 2: ResNet18 (Only top 2 layers changed and trained) 
 - Number of epochs = 2
 - **Train Accuracy: 96.77%**
 - **Validation Accuracy: 96.88%**

- Model 3: ResNet50 (Only top 2 layers changed and trained) 
 - Number of epochs = 2
 - **Train Accuracy: **
 - **Validation Accuracy: **
