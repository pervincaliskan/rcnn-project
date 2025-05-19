import torch
import torch.nn as nn
import torchvision.models as models


def get_backbone(name='resnet50', pretrained=True):
    """
    CNN omurga modeli oluşturur

    Args:
        name (str): Omurga model adı ('resnet18', 'resnet34', 'resnet50', 'vgg16')
        pretrained (bool): Önceden eğitilmiş ağırlıkları kullan

    Returns:
        model: Omurga model
        feature_dim: Özellik vektörü boyutu
    """
    feature_dim = 0

    if name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        feature_dim = 512
    elif name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        feature_dim = 512
    elif name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        feature_dim = 2048
    elif name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        feature_dim = 4096
    else:
        raise ValueError(f"Desteklenmeyen omurga modeli: {name}")

    # Son sınıflandırma katmanını kaldır
    if 'resnet' in name:
        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)
    elif name == 'vgg16':
        model = nn.Sequential(*list(model.features.children()))

    return model, feature_dim