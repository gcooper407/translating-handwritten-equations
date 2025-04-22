import torch.nn as nn
import torch.nn.functional as F

#The CNN function that recieves photos of subsections of the hand wirtten equation
# This CNN function only recieves what is meant to be an image of one symbol or a small part of the equation
# Uses 2 convlutional layers with ReLU and pooling
# Uses 1 output layer 

#define the model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        #3 channels, 32 feature maps, kernel size of 3, padding for spacial size
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        #reduce dimensions by 1/2
        self.pool = nn.MaxPool2d(2)

        #flat
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    #run model
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)