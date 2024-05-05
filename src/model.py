import torch
import torch.nn as nn
import torch.nn.functional as F

class CalligraphyCNN(nn.Module):
    def __init__(self):
        super(CalligraphyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Padding to maintain size
        self.pool1 = nn.MaxPool2d(2, 2)  # Output size: 32@32x32
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output size: 64@32x32
        self.pool2 = nn.MaxPool2d(2, 2)  # Output size: 64@16x16
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output size: 128@16x16
        self.pool3 = nn.MaxPool2d(2, 2)  # Output size: 128@8x8
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Output size: 256@8x8
        self.pool4 = nn.MaxPool2d(2, 2)  # Output size: 256@4x4
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)  # Output 4 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        
        x = x.view(-1, 256 * 4 * 4)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)