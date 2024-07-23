from vgg16_model import VGG16
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from load_data import train_loader

model = VGG16()

feature_list = []

for batch_id, (images, labels) in enumerate(train_loader):
    if (batch_id == 0):
        images = images.view(3, images.size(0), 224, 224)
        output_feature = model(images)
        feature_list.append(output_feature)
    
print(feature_list)