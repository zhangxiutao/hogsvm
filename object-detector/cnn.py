import torch
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 5)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32*5*20, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 2)
        self.softmax = nn.Softmax(dim=1)

        
    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        #print(self.conv1(x).size()) #(32, 16, 31, 91) (batch, depth, width, height)
        x = self.pool(F.relu(self.conv1(x)))
        #print(x.size()) #(32, 16, 15, 45) 
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.size()) #(32, 32, 5, 20)
        x = self.dropout(x)
        x = x.view(-1, 32*5*20)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        #print(self.fc3(x))
        x = self.softmax(self.fc3(x))
        #print(x)
        return x

