import microtorch.nn as nn
import microtorch.nn.functional as F

class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        # input size of MNIST (Batch, 1, 28, 28)
        self.conv1 = nn.Conv2D(1, 16, kernel_size=3, stride=2, padding=1) # (batch, 16, 14, 14)
        self.conv2 = nn.Conv2D(16, 32, kernel_size=3, stride=2, padding=1) # (batch, 32, 7, 7)
        self.conv3 = nn.Conv2D(32, 64, kernel_size=3, stride=2, padding=1) # (batch, 64, 4, 4)
        self.linear1 = nn.Linear(1024, 128) # (batch, 128)
        self.linear2 = nn.Linear(128, 10) # (batch, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # flatten the feature map (batch, 64, 4, 4) -> (batch, 1024)
        x = x.reshape(x.shape[0], -1)

        x = F.relu(self.linear1(x))
        return self.linear2(x)