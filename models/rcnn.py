import torch
import torch.nn as nn
from models.backbone import get_backbone


class RCNN(nn.Module):
    def __init__(self, num_classes, backbone_name='resnet50', pretrained=True):
        super(RCNN, self).__init__()

        # CNN omurga
        self.backbone, feature_dim = get_backbone(backbone_name, pretrained)

        # Özellik boyutunu azaltma (isteğe bağlı)
        self.feature_reducer = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # Sınıflandırma katmanı
        self.classifier = nn.Linear(512, num_classes + 1)  # +1 arka plan sınıfı için

    def forward(self, x):
        # Omurga modelinden özellikler çıkar
        x = self.backbone(x)
        x = torch.flatten(x, 1)

        # Özellik boyutunu azalt
        features = self.feature_reducer(x)

        # Sınıflandırma
        scores = self.classifier(features)

        return features, scores