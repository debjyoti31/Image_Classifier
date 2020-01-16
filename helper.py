from matplotlib import pyplot as py

import torch
from torch import nn, optim
import torch.nn.functional as f
from torchvision import datasets, transforms, models
from PIL import Image
import os, random
import seaborn as sns
import numpy as np

import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

arc = {"resnet50":2048,
        "vgg16":25088,
        "alexnet":9216}

def load_data(from_loc  = "./flowers" ):
    
    data_dir = from_loc
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir,transform=test_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size =64)
    
    return trainloader , testloader, validationloader,train_data


def netwk_setup(structure, dropout, lr,hidden_units):

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = models.resnet50(pretrained=True)
    if structure == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained=True) 
        
    else:
        ("The models are limited to resnet50,vgg16 and alexnet only. ")

    for param in model.parameters():
        param.required_grad = False

    classifier = nn.Sequential(nn.Linear(arc[structure],hidden_units),
                                  nn.ReLU(),
                                  nn.Dropout(p=dropout),
                                  nn.Linear(hidden_units,102),
                                  nn.LogSoftmax(dim = 1))
    
    if structure == 'resnet50':
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr)
    elif structure == 'vgg16' or 'alexnet':
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    criterion = nn.NLLLoss()
    
    model.to(device);
    
    return model, criterion, optimizer


def train_network(model,criterion,optimizer,trainloader,testloader,epochs):

    #epochs = 4
    steps = 0
    running_loss = 0
    ascent = 4
    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % ascent == 0:
                testloss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in testloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)

                        testloss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/ascent:.3f}.. "
                      f"test loss: {testloss/len(testloader):.3f}.. "
                      f"test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()
                
    print("TRAINING HAS BEEN COMPLETED")
    
def validation_data_set(model,validationloader,criterion):
    validationloss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for images, labels in validationloader:
            images, labels = images.to(device), labels.to(device)
            logps = model.forward(images)
            batch_loss = criterion(logps, labels)

            validationloss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"validation_data_set accuracy is: {100 *(accuracy/len(validationloader)):.3f}")

def save_checkpoint(model,structure,hidden_units,dropout,lr,epochs,optimizer,checkpoint,from_loc  = "./flowers"):
    
    _,_,_,train_data = load_data(from_loc  = "./flowers" )
    model,_,_ = netwk_setup(structure, dropout, lr,hidden_units)

    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'structure' :structure,
                'hidden_units':hidden_units,
                'dropout':dropout,
                'lr':lr,
                'epochs':epochs,
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'class_to_idx':model.class_to_idx}
    
    model.cpu()
    torch.save(checkpoint,"checkpoint.pth")


def load_check_point(checkpoint):
    checkpoint = torch.load(checkpoint)

    structure = checkpoint['structure']
    hidden_units = checkpoint['hidden_units']
    dropout = checkpoint['dropout']
    lr=checkpoint['lr']

    model,_,_ = netwk_setup(structure, dropout, lr,hidden_units)
    for param in model.parameters():
        param.required_grad = False
        
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image_path):

    img = Image.open(str(image_path))
    adjust = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    img_tensor = adjust(img)
    
    return img_tensor

def predict(image_path, model,device,topk=5):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():    
        pro_image = process_image(image_path)
        pro_image.unsqueeze_(0)
        pro_image = pro_image.float()
        logps = model(pro_image.to(device))
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)

        idx_to_class = {}
        for key, value in model.class_to_idx.items():
            idx_to_class[value] = key

        top_class_np = top_class[0].cpu().numpy()

        top_labels = []
        for label in top_class_np:
            top_labels.append(int(idx_to_class[label]))

    return top_p, top_labels

