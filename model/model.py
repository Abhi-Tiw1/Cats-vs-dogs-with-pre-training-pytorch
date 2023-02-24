import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch
from torchvision.models import resnet50, resnet18

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Mnist_Experimental(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 250)
        self.fc2 = nn.Linear(250, 50)
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(F.max_pool2d(self.conv1(x),2, 2)))
        x = F.relu(self.bn2(F.max_pool2d(self.conv2(x),2, 2)))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
class MNIST_RNN(nn.Module):
    """
    Args gets values:
        "input_size": 112,
        "hidden_size": 128,
        "num_layers": 2,
        "num_classes": 10,
        "sequence_length": 28
    """
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length):
        super(MNIST_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.seq_length = sequence_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,bidirectional = True, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.device = 'cpu'
        
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        x = x.squeeze()
        x = x.reshape([x.size(0),7,112])
        # Passing in the input and hidden state into the model and  obtaining outputs
        out, hidden = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        #Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out[:, -1, :])
        return out
       

class CatAndDogConvNet(nn.Module):

    def __init__(self):
        super().__init__()

        # onvolutional layers (3,16,32)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=(5, 5), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=(5, 5), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=(3, 3), padding=1)
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        
        # conected layers
        self.fc1 = nn.Linear(in_features= 64 * 3 * 3, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=2)


    def forward(self, X):

        X = F.relu(self.bn1(self.conv1(X)))
        X = F.max_pool2d(X, 2)

        X = F.relu(self.bn2(self.conv2(X)))
        X = F.max_pool2d(X, 2)

        X = F.relu(self.bn3(self.conv3(X)))
        X = F.max_pool2d(X, 2)
        
        X = F.relu(self.bn4(self.conv4(X)))
        X = F.max_pool2d(X, 2)

        X = X.view(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return X


class ResNet50(nn.Module):

    def __init__(self, pretrained):
        super().__init__()
        
        self.pretrained = pretrained
        #load the model
        self.model = resnet50(pretrained = self.pretrained)
        
        # freeez weights of all layers
        for p in self.model.parameters():
            p.requires_grad  = False
        
        # change final layer with two layers
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 64, bias = True),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 2, bias = True),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self,X):
        return self.model(X)
        

class ResNet18(nn.Module):

    def __init__(self, pretrained):
        super().__init__()
        
        self.pretrained = pretrained
        #load the model
        self.model = resnet18(pretrained = self.pretrained)
        
        # freeez weights of all layers
        for p in self.model.parameters():
            p.requires_grad  = False
        
        # change final layer with two layers
        self.model.fc = nn.Sequential(
            nn.Linear(512, 2, bias = True),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self,X):
        return self.model(X)
        
