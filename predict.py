import matplotlib.pyplot as py
import numpy as np
import torch
from torch import nn, tensor,optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import PIL
from PIL import Image
import json
import os, random
import argparse

import helper

arg = argparse.ArgumentParser(description='predict.py')

#Command Line Arguments
arg.add_argument('input_img', default='/home/workspace/ImageClassifier/flowers/test/1/image_06752.jpg', 
                 nargs='*', action="store", type = str)
arg.add_argument('checkpoint', default='/home/workspace/ImageClassifier/checkpoint.pth', nargs='*', action="store",type = str)
arg.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
arg.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
arg.add_argument('--gpu', dest="gpu", action="store", default='device')
arg.add_argument('--model', dest="arch", action="store", default="resnet50", type = str)
arg.add_argument('--catagory_name', dest="catagory_name", action="store", default="cat_to_name.json", type = str)

pa_arg = arg.parse_args()
image_path = pa_arg.input_img
from_loc = pa_arg.data_dir
topk = pa_arg.top_k
model = pa_arg.arch
dev = pa_arg.gpu
checkpoint = pa_arg.checkpoint
#cat_name = pa_arg.catagory_name


#trainloader , testloader, validationloader, train_data = helper.load_data(from_loc)
model = helper.load_check_point(checkpoint)
    
    
top_p, top_labels = helper.predict(image_path, model,dev,topk)

#with open('cat_name', 'r') as f:
        #cat_to_name = json.load(f)
if pa_arg.catagory_name:
    with open(pa_arg.catagory_name, 'r') as f:
        cat_to_name = json.load(f)
        print("Category names have been loaded successfully...")
        
top_flowers_names = [cat_to_name[str(lab)] for lab in top_labels]

print(f"The top 5 classes are {top_p}\n",
      f"The top 5 labels are {top_labels}\n",
     f"The top 5 flowers are {top_flowers_names}")