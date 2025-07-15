import torch
import torch.nn as nn
import torch.nn.functional as F
class StudentVisionEagle(nn.Module):
    def __init__(self, num_classes=6):
        super(StudentNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3, stride=1,
                     padding=1),  # [B, 3, 100, 100] -> [B, 32, 100, 100]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> [B, 32, 50, 50]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> [B, 64, 50, 50]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [B, 64, 25, 25]
        
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # -> [B, 128, 25, 25]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # -> [B, 128, 1, 1]
        )
        self.classifier= nn.Sequential(
            nn.Flatten(), # -> [B,128]
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            
        )
    def forward(self,x):
        x=self.features(x)
        x=self.classifier(x)
        return x
        
