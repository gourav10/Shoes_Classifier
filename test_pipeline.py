import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets,models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from engine import Engine

def imshow(inp, title = None):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    inp = std*inp+mean
    inp = np.clip(inp,0,1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(5)

def visualize_data(dataloader,brand_list):
    # data_iter = iter(dataloader)
    
    inputs, labels   = next(iter(dataloader))
    out = torchvision.utils.make_grid(inputs)

    imshow(out,title=[brand_list[x] for x in labels])


def load_model(device):
    model = torch.jit.load('model_scripted.pt')
    return model

cudnn.benchmark = True
plt.ion() #interactive mode

data_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]
)


test_dataset = datasets.ImageFolder(os.path.join('data','test'),data_transforms)
test_dataloader = DataLoader(test_dataset,batch_size=4,shuffle=True)

brand_list = test_dataset.classes

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  

dataset_size = len(test_dataset)

print(brand_list,dataset_size)
# visualize_data(test_dataloader,brand_list)

# Load Pre-trainied model
model_ft = load_model(device)

# print(model_ft)

engine = Engine()

engine.test(model=model_ft,
            dataloader= test_dataloader,
            dataset_size=dataset_size,
            device=device
            )




