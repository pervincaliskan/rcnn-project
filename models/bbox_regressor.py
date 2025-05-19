import torch
import torch.nn as nn


class BBoxRegressor(nn.Module):
    def __init__(self, input_dim=512):
        super(BBoxRegressor, self).__init__()

        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 4)  # tx, ty, tw, th koordinatları
        )

    def forward(self, features):
        return self.regressor(features)