import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from dataset import Dataset
# device = torch.cpu.current_device('cpu')

training_data_directory = 'extracted_frames'

BATCH_SIZE = 64

transform_train = transforms.Compose([transforms.Resize((224, 224)), 
                                      transforms.ToTensor(), 
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = Dataset(
    root=training_data_directory, 
    transform=transform_train,
    target_transform=None)

NUM_EPOCHS = 50

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# train_features, train_labels, train_path = next(iter(train_loader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# img = torchvision.transforms.functional.to_pil_image(img)
# label = train_labels[0]
# path = train_path[0]
# print(f"Label: {label}")
# print(path)
# plt.imshow(img, cmap="gray")
# plt.show()


