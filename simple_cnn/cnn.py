from torch import nn  # All neural network modules
import torch.nn.functional as F # For activation functions


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,  # Number of filters
            kernel_size=3,  # Size of the filter
            stride=1,   # How much we move our filter
            padding=1,  # To keep the same size of the image
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8,  # Input from the previous layer
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)  # 7x7 image dimension

    def forward(self, x):
        """
        Defines the forward pass of the CNN.
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width). Initially x.shape = (64, 1, 28, 28). 
        Returns:
            torch.Tensor: Output tensor after passing through the network layers.
        """
        x = F.relu(self.conv1(x))  # x.shape = (64, 8, 28, 28)
        x = self.pool(x)
        x = F.relu(self.conv2(x))  # x.shape = (64, 16, 7, 7)
        x = self.pool(x)
        # Flatten the image. x.shape = (64, 16*7*7)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)  # x.shape = (64, 10)
        return x
