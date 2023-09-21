import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1),

            nn.Conv2d(32, 64, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1),

            nn.Conv2d(64, 128, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1),

            nn.Conv2d(128, 128, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1),

            nn.Dropout(0.3)
        )

        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.feature_extraction(input)
        output = output.view(output.size(0), -1)
        output = self.classification(output)

        return output

