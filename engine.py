from cv2 import phase
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


class Engine:
    def __init__(self) -> None:
        pass

    def set_learning_rate(self,lr=0.01):
        self.learn_rate = lr

    def __validate(self, model,dataloader, dataset_size, device, criterion, optimizer, scheduler, epoch,num_epochs = 25):
        model.eval()
        
        running_loss = 0
        running_corrects = 0

        for batch_idx, (inputs,labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _,preds = torch.max(outputs,1)
                loss = criterion(outputs, labels)

            running_loss +=loss.item()*inputs.size(0)
            running_corrects += torch.sum(preds==labels.data)
            # print(f'[VALIDATE BATCH ID: {batch_idx}] Loss: {self.running_loss:.4f} Accuracy: {self.running_corrects:.4f}')

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects / dataset_size

        print(f'[VALIDATE][{epoch}/{num_epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}')
        return epoch_loss,epoch_acc       


    def __train(self, model, train_dl, dataset_size, device, criterion, optimizer, scheduler, epoch,num_epochs = 25):
        model.train()

        running_loss = 0
        running_corrects = 0

        for batch_id, (inputs,labels) in enumerate(train_dl):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _,preds = torch.max(outputs,1)
                
                loss = criterion(outputs,labels)

                loss.backward()
                optimizer.step()
            
            running_loss +=loss.item()*inputs.size(0)
            running_corrects += torch.sum(preds==labels.data)
            # print(f'[TRAIN BATCH ID: {batch_id}] Loss: {running_loss:.4f} Accuracy: {running_corrects:.4f}')
            scheduler.step()

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects / dataset_size

        print(f'[TRAIN][{epoch}/{num_epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}')
        return epoch_loss,epoch_acc


    def train_model(self, model, train_dataloader, val_dataloader, dataset_sizes, device, criterion, optimizer, scheduler, num_epochs = 25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())

        best_acc = 0.0

        self.running_loss = 0.0
        self.running_corrects = 0

        for epoch in range(num_epochs):

            print(f'Epoch {epoch}/{num_epochs}')
            print('-'*10)
            
            self.__train(model,train_dataloader, dataset_sizes['train'], device, criterion, optimizer, scheduler, epoch, num_epochs)
            
            if epoch%5 == 0:
                _,val_acc = self.__validate(model, val_dataloader, dataset_sizes['val'], device, criterion, optimizer, scheduler, epoch, num_epochs)

                if (val_acc>best_acc):
                    best_acc = val_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f'Saving model with Accuracy: {best_acc:.4f}')
                    model_scripted = torch.jit.script(model)
                    model_scripted.save('model_scripted.pt')
                    # torch.save(model.state_dict(),'model_scripted.pt')
        print()
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val accuracy: {best_acc:.4f}')

        model.load_state_dict(best_model_wts)
        return model

    def test(self, model=None, dataloader=None,dataset_size=None,device = None, class_labels = []):
        print("Inside in the Test Method")
        model.eval()
        model.to(device)

        ovr_correct = 0
        total_items = 0

        for batch_idx, (inputs,labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            total = labels.size(0)
            

            with torch.no_grad():
                outputs = model(inputs)
                _,preds = torch.max(outputs,1)
                predicted_results = preds.detach().to('cpu').numpy()
                # loss = criterion(outputs, labels)
                correct = (preds == labels).sum()
                ovr_correct+=correct
                total_items+=total
                acc = 100*correct/total
                # print(f'{acc:.2f}')
                print("Result: {}, Inference Accuracy: {:.2f}".format(predicted_results,acc))
                # conf_score, class_idx = torch.max(preds)

                # predicted_class = class_labels[class_idx]
                # print(f"Class: {predicted_class}, Confidence: {round(conf_score*100,2):.2f}")
        ovr_acc = 100*ovr_correct/total_items
        print("Overall Model Accuracy: {:.2f}".format(ovr_acc))