# EfficientNet
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
class EffiNet(nn.Module):
    def __init__(self):
        super(EffiNet, self).__init__()
        self.base_model = EfficientNet.from_pretrained("efficientnet-b4")
        num_ftrs = self.base_model._fc.in_features
        self.base_model._fc = nn.Linear(num_ftrs,4, bias = True)
        
    def forward(self, image):
        out = self.base_model(image)
        return out
# Resnet50
import torchvision
class Resnet50(nn.Module):
    def __init__(self,model_no):
        super(Resnet50, self).__init__()
        self.base_model = torchvision.models.resnext50_32x4d(pretrained=True)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs,4, bias = True)
        
    def forward(self, image):
        out = self.base_model(image)
        return out    
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
class EffiNet7(nn.Module):
    def __init__(self):
        super(EffiNet7, self).__init__()
        self.base_model = EfficientNet.from_pretrained("efficientnet-b7")
        num_ftrs = self.base_model._fc.in_features
        self.base_model._fc = nn.Linear(num_ftrs,4, bias = True)
        
    def forward(self, image):
        out = self.base_model(image)
        return out 
# densenet
import torchvision
class Densenet(nn.Module):
    def __init__(self,model_no):
        super(Densenet, self).__init__()
        self.base_model = torchvision.models.densenet161(pretrained=True)
        num_ftrs = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(num_ftrs,4, bias = True)
        
    def forward(self, image):
        out = self.base_model(image)
        return out  