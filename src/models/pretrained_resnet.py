from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetPT(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.c1 = nn.Conv2d(3, 16, stride=1, padding=1, kernel_size=3)
        self.c2 = nn.Conv2d(16, 32, stride=1, padding=1, kernel_size=3)
        self.c3 = nn.Conv2d(32, 64, stride=1, padding=1, kernel_size=3)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adap_pool = nn.AdaptiveMaxPool2d(output_size=32)

        self.fc1 = nn.Linear(64 * 32 * 32, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 512)
        self.fc4 = nn.Linear(512, 2)

    def forward(self, x: torch.tensor) -> Any:
        x = F.relu(self.c1(x))
        x = self.max_pool(x)
        x = F.relu(self.c2(x))
        x = self.max_pool(x)
        x = F.relu(self.c3(x))
        x = self.adap_pool(x)
        x = x.view(-1, 64 * 32 * 32)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)

        return x
