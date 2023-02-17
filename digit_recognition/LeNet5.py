from collections import OrderedDict
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        # 卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
        )
        # 全连接层
        self.fc = nn.Sequential(
            
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
    # 前向传播
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output