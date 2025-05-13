import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, hidden_units=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        conv_output = 32 * 32 * 32
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)