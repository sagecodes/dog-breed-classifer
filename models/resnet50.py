import torchvision.models as models
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.transforms as transforms


import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image


# TODO refactor for model type?

# Base model
    # init

# forward (build)
#   break
#  
# Save model
#
# Load model
#
# Log function
#


class Resnet50_pretrained:
    
    def __init__(self, num_classes, use_cuda=False):

        self.use_cuda = use_cuda

        # use pre-trained resnet50 model from pytorch
        self.model = models.resnet50(pretrained=True)

        # freeze model params for features
        for param in self.model.parameters():
            param.requires_grad = False

        # configure output layer for number of classes
        self.model.fc = nn.Linear(2048,
                            num_classes,
                            bias=True)

        if self.use_cuda:
            self.model = self.model.cuda()
        
    def build(self):

        return self.model

    def save(self):
        '''Save model'''
        pass

    def load():
        '''load model weights'''
        pass

    def log():
        '''Training & Validation logs '''
        pass