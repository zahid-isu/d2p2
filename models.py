import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from torchvision.models import resnet18


# Linear model.
class LinearModel(nn.Module):
    """Linear model class."""
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = self.fc(x)
        return x


# CNN model. Note we only use this for CIFAR such that the architecture is
# specific. If with MNIST type of data, it may require some minor changes.
# Hardcode some values here for the architecture.
class ConvolutionalNeuralNet(nn.Module):
    """Convolutional neural network model."""
    def __init__(self):
        super(ConvolutionalNeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) #for cifar channel=3, for mnist channel=1
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4* 4, 120) # for cifar (16*5*5), for mnist (16*4*4) for usps(16*1*1)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Multilayer perceptron.
# Hardcode some values here for the architecture.
class MultilayerPerceptron(nn.Module):
    """Multilayer perceptron model."""
    def __init__(self):
        super(MultilayerPerceptron, self).__init__()
        self.linear1 = nn.Linear(784, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return F.log_softmax(x, dim=1)


# Build custom module for logistic regression.
# Hardcode some values here for the model.
class LogisticRegression(torch.nn.Module):
    """Logistic regression model."""
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(784, 10)
    # make predictions
    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        return x

# DenseNet Model.
# Hardcode some values here for the architecture.

class BasicBlockD(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlockD, self).__init__()
        self.bn1 = nn.GroupNorm(num_groups=min(32, in_planes // 4), num_channels=in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.GroupNorm(num_groups=min(32, in_planes // 4), num_channels=in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.GroupNorm(num_groups=min(32, in_planes // 4), num_channels=in_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.GroupNorm(num_groups=min(32, in_planes // 4), num_channels=in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=False, dropRate=0.0):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlockD
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.GroupNorm(num_groups=min(32, in_planes // 4), num_channels=in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        # Ensure num_groups is a divisor of out_channels
        num_groups = min(8, out_channels)  # Choose a fixed small number of groups that divides most common channel sizes
        while out_channels % num_groups != 0:  # Adjust num_groups if it does not divide out_channels
            num_groups -= 1

        self.bn1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_channels = 16

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(16)
        self.bn1 = nn.GroupNorm(num_groups=min(32, 16 // 4), num_channels=16)
        self.relu = nn.ReLU(inplace=True)

        # Create 5 layers ResNet20
        self.layer1 = self._make_layer(16, 5, stride=1)
        self.layer2 = self._make_layer(32, 5, stride=2)
        self.layer3 = self._make_layer(64, 5, stride=2)
        self.layer4 = self._make_layer(128, 2, stride=2) 
        self.layer5 = self._make_layer(256, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)  # First block stride is 2, others are 1
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

class ResNet20Small(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20Small, self).__init__()
        self.in_channels = 16  # Set initial channels to 16

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)  # Adjusted to 16
        self.bn1 = nn.GroupNorm(num_groups=4, num_channels=16)  # Adjusted to use 4 groups for 16 channels
        self.relu = nn.ReLU(inplace=True)

        # Adjusted layers to make them closer to ResNet20 but slightly smaller
        self.layer1 = self._make_layer(16, 4, stride=1)  # Adjusted to 4 blocks
        self.layer2 = self._make_layer(32, 4, stride=2)  # Adjusted channels and blocks
        self.layer3 = self._make_layer(64, 3, stride=2)  # Slightly fewer channels

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)  # Adjusted to match the final number of channels

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # First block stride is as specified, others are 1
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

class ResNet20Medium(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20Medium, self).__init__()
        self.in_channels = 32

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=32)  # Adjust number of groups for GroupNorm
        self.relu = nn.ReLU(inplace=True)

        # Adjusted layers to increase the number of parameters
        self.layer1 = self._make_layer(32, 6, stride=1)  # Increased to 6 blocks
        self.layer2 = self._make_layer(64, 6, stride=2)  # Increased to 6 blocks and doubled channels
        self.layer3 = self._make_layer(128, 5, stride=2)  # Increased to 5 blocks

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)  # Adjusted to match the final number of channels

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # First block stride is as specified, others are 1
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# Model types
ModelType = typing.Union[
    LinearModel,
    ConvolutionalNeuralNet,
    MultilayerPerceptron,
    LogisticRegression,
    DenseNet3,
    ResNet20
]