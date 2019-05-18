import torch.nn as nn
import torch.nn.functional as F


class CapsuleNetwork(nn.Module):
    def __init__(self):
        super(CapsuleNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, 9)
        self.conv2 = nn.Conv2d(256, 32 * 8, 9, stride=2)
        self.conv3 = nn.Conv2d(256, 10*16, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)

        return x.view(-1, 10, 16)
