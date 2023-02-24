# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 18:16:36 2022

@author: Abhishek
"""

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np

class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # input feature with batch first is true 
        # --> batch = 100, seq size = 28. features  = 28 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        pass
    
    def forward(self, x):
        
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Passing in the input and hidden state into the model and  obtaining outputs
        out, hidden = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        #Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out[:, -1, :])
        return out
       
        pass
pass

def train(num_epochs, model, loaders):
    # Train the model
    total_step = len(loaders['train'])
       
    print(f"num_epochs: {num_epochs}")
    print(f"model: {model}")
    print(f"loaders['train']: {loaders['train']}")
    print("Started epoch: ")
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            #print("epoch: ", epoch)
            #print(f"images: {images}")
            #print(f"labels: {labels}")
            #print(images.size())
            model.train()
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            # print(f"images after reshpae: {images}")
            
            # Forward pass
            outputs = model(images)
            loss = loss_func(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            
            pass
        
        pass
    print("Ended epoch: ")
    pass

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)

# print(train_data)
# print(train_data.data.size())
# print(train_data.targets.size())



loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
}



sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 1
num_classes = 10
batch_size = 64
num_epochs = 5
learning_rate = 0.01

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)


loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)   
train(num_epochs, model, loaders)
num_params = 0
for p in model.parameters():
    if p.requires_grad:  
        num_params = num_params + np.prod(p.size())
print('# Paramters: ', num_params)
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in loaders['test']:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total = total + labels.size(0)
        correct = correct + (predicted == labels).sum().item()
print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))