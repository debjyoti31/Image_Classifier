import matplotlib.pyplot as py
import numpy as np
import torch
from torch import nn, tensor,optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse

import helper

arg = argparse.ArgumentParser(description='Train.py')
# Command Line ardguments

arg.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
arg.add_argument('--gpu', dest="gpu", action="store", default='device')
arg.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
arg.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.003)
arg.add_argument('--dropout', dest = "dropout", action = "store", default = 0.2)
arg.add_argument('--epochs', dest="epochs", action="store", type=int, default=4)
arg.add_argument('--arc', dest="arc", action="store", default="resnet50", type = str)
arg.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=512)

pa_arg = arg.parse_args()

from_loc = pa_arg.data_dir
checkpoint = pa_arg.save_dir
lr = pa_arg.learning_rate
structure = pa_arg.arc
dropout = pa_arg.dropout
dev = pa_arg.gpu
epochs = pa_arg.epochs
hidden_units = pa_arg.hidden_units


trainloader , testloader, validationloader,train_data = helper.load_data(from_loc)
model, criterion, optimizer = helper.netwk_setup(structure, dropout, lr,hidden_units)

helper.train_network(model,criterion,optimizer,trainloader,testloader,epochs)
helper.validation_data_set(model,validationloader,criterion)

helper.save_checkpoint(model,structure,hidden_units,dropout,lr,epochs,optimizer,checkpoint,from_loc)