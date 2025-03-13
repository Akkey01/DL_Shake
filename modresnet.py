import torch
import torch.nn as nn
import torch.nn.functional as F

###########################################
# Original BasicBlock (kept for reference)
###########################################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(p=0.3)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

###########################################
# Shake-Shake Regularization Implementation
###########################################
class ShakeShakeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, training):
        if training:
            alpha = torch.rand(x1.size(0), 1, 1, 1, device=x1.device)
        else:
            alpha = 0.5
        ctx.save_for_backward(alpha)
        return x1 * alpha + x2 * (1 - alpha)

    @staticmethod
    def backward(ctx, grad_output):
        alpha, = ctx.saved_tensors
        beta = torch.rand(alpha.size(), device=grad_output.device)
        return grad_output * beta, grad_output * (1 - beta), None

###########################################
# Shake-Shake Block: Two-branch residual block
###########################################
class ShakeShakeBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ShakeShakeBlock, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = ShakeShakeFunction.apply(out1, out2, self.training)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

###########################################
# Shake-Shake ResNet for CIFAR-10
# This network uses [4,4,4] blocks with an initial width of 32.
# Total parameter count ~2.9M (< 5M).
###########################################
class ShakeShakeResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, base_channels=32):
        super(ShakeShakeResNet, self).__init__()
        self.in_planes = base_channels
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.layer1 = self._make_layer(block, base_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 4 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Factory function to create a Shake-Shake ResNet-26 (depth = 6*4+2 = 26 layers)
def ShakeShakeResNet26(num_classes=10):
    return ShakeShakeResNet(ShakeShakeBlock, [4, 4, 4], num_classes=num_classes, base_channels=32)

###########################################
# (Optional) Original ModResNet18 remains available for reference.
###########################################
class ModResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ModResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 208, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 288, num_blocks[3], stride=2)
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(288 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out

def ModResNet18():
    return ModResNet(BasicBlock, [2, 2, 2, 2])
