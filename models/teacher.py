import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VisionEagle(nn.Module):
    def __init__(self, num_classes=6):
        super(VisionEagle, self).__init__()

        # Load two ResNet18 backbones
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18_2 = models.resnet18(pretrained=True)

        # Use shared feature extractors instead of repeating manually
        self.stem1 = nn.Sequential(
            self.resnet18.conv1,
            self.resnet18.bn1,
            self.resnet18.relu,
            self.resnet18.maxpool
        )

        self.stem2 = nn.Sequential(
            self.resnet18_2.conv1,
            self.resnet18_2.bn1,
            self.resnet18_2.relu,
            self.resnet18_2.maxpool
        )

        # ResNet layers
        self.layer1 = self.resnet18.layer1
        self.layer2 = self.resnet18.layer2
        self.layer3 = self.resnet18.layer3
        self.layer4 = self.resnet18.layer4

        self.layer12 = self.resnet18_2.layer1
        self.layer22 = self.resnet18_2.layer2
        self.layer32 = self.resnet18_2.layer3
        self.layer42 = self.resnet18_2.layer4

        # Attention modules
        self.scan_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.scan_attention = nn.Conv2d(128, 1, kernel_size=1)

        self.scan_conv22 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.scan_attention2 = nn.Conv2d(128, 1, kernel_size=1)

        self.scan_conv23 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.scan_attention3 = nn.Conv2d(256, 1, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # First stem (for attention map 1)
        x1 = self.stem1(x)
        scan_out = F.relu(self.scan_conv2(x1))
        attention_map = torch.sigmoid(self.scan_attention(scan_out))
        attention_map = F.interpolate(attention_map, size=x.size()[2:], mode='bilinear', align_corners=False)
        x = x * attention_map

        # Second stem with attention-applied input
        x = self.stem2(x)

        # First residual + attention block
        x1 = self.layer1(x)
        scan_out = F.relu(self.scan_conv22(x1))
        attention_map = torch.sigmoid(self.scan_attention2(scan_out))
        attention_map = F.interpolate(attention_map, size=x.size()[2:], mode='bilinear', align_corners=False)
        x = x * attention_map

        # Second residual + attention block
        x1 = self.layer2(x)
        scan_out = F.relu(self.scan_conv23(x1))
        attention_map = torch.sigmoid(self.scan_attention3(scan_out))
        attention_map = F.interpolate(attention_map, size=x.size()[2:], mode='bilinear', align_corners=False)
        x = x * attention_map

        # Final residual layers
        x = self.layer22(x)
        x = self.layer32(x)
        x = self.layer42(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
