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

cudnn.benchmark = True
plt.ion() #interactive mode

data_transforms = {
    'train':transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    'val':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
}


train_dataset = datasets.ImageFolder(os.path.join('data','train'),data_transforms['train'])
train_dataloader = DataLoader(train_dataset,batch_size=4,shuffle=True)

brand_list = train_dataset.classes

print(brand_list)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  

# visualize_data(train_dataloader,brand_list)


validation_dataset = datasets.ImageFolder(os.path.join('data','validation'),data_transforms['train'])
validation_dataloader = DataLoader(validation_dataset, batch_size=4,shuffle=True)

brand_list = validation_dataset.classes

dataset_size = {
    'train': len(train_dataset),
    'val':len(validation_dataset)
}

# visualize_data(validation_dataloader,brand_list)

# Load Pre-trainied model
model_ft = models.resnet18(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False

num_filters = model_ft.fc.in_features

# Add Linear layer for binary classification
model_ft.fc = nn.Linear(num_filters,2)
model_ft = model_ft.to(device=device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr = 0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7,gamma=0.1)

engine = Engine()

res_model = engine.train_model(model= model_ft,
                            train_dataloader=train_dataloader,
                            val_dataloader=validation_dataloader,
                            dataset_sizes=dataset_size,
                            device=device,
                            criterion=criterion,
                            optimizer= optimizer_ft,
                            scheduler=exp_lr_scheduler,num_epochs=25)


