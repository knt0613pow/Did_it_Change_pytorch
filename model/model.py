import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision import models
import torch
import numpy as np
from cirtorch.networks.imageretrievalnet import init_network

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        
class Resnet_feature(BaseModel):
    def __init__(self):
        super().__init__()
        model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*(list(model.children())[:-1]))
        
    def forward(self, x):
        return self.model(x)



class cnn_IR(BaseModel):
    def __init__(self):
        super().__init__()
        model_params = {}
        model_params['architecture'] = 'resnet101'
        model_params['pooling'] = 'gem'
        model_params['local_whitening'] = True
        model_params['regional'] = True
        model_params['whitening'] = True
        model_params['pretrained'] = True
        model = init_network(model_params)
        self.model = model
        
    
    def forward(self, x):
        return self.model(x)
