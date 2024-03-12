import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN_Imagenet(nn.Module):
    def __init__(self):
        super(SimpleCNN_Imagenet, self).__init__()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(256*56*56, 128)
        self.fc = nn.Linear(128, 128)
        self.output = nn.Linear(128, 10)
        
    def forward(self, x):
        B = x.size()[0]

        x = F.relu(self.conv1(x))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.max_pool(F.relu(self.conv4(x)))
        x = F.relu(self.linear(x.view(B, 256*56*56)))
        x = F.dropout(F.relu(self.fc(x)), 0.5, training=self.training)
        x = self.output(x)

        return x
    
    def loss(self, pred, label):
        return F.cross_entropy(pred, label)
