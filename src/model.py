import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, ):
        super(CNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32 * 56 * 56)
        x = self.classifier(x)
        return x