from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
import scipy.io as scio
from PIL import Image 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import random
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Dataset settings
data_dir = 'door_val_0309/'

# Model settings
model_dir = 'nets/vgg_ft.pkl'

# Number of classes in the dataset
num_classes = 7

# Batch size for training (change depending on how much memory you have)
batch_size = 32

def output_test(file, names, labels, result, outputs):
    for i in range(len(names)):
        # visualize txt
        content = "name:{}  gt:{}  prediction:{}  raws:[{}, {}, {}, {}]\n".format(names[i], labels[i], result[i], str(int(round(outputs[i][0]*-40))), \
        str(int(round(outputs[i][1]*40))), str(int(round(outputs[i][2]*-40))), str(int(round(outputs[i][3]*40))))
        
        file.write(content)

def output_aggregate(outputs):
    result = []
    for i in range(len(outputs)):
        result.append([(int(round(outputs[i][0]*-40)) not in range(-4,1)) or (int(round(outputs[i][1]*40)) not in range(0,5)) or (int(round(outputs[i][2]*-40)) not in range(-4,1))\
        or (int(round(outputs[i][3]*40)) not in range(0,5))])

    return torch.FloatTensor(result)

def test_model(model, dataloaders, criterion):
    since = time.time()
    file = open("./test_out.txt",'w')
    
    running_loss = 0
    for names, inputs, labels in dataloaders:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        model.eval()
        
        outputs = model(inputs)
        aggrs = output_aggregate(outputs.cpu().detach().numpy())
        aggrs = aggrs.to(device)
        loss = criterion(aggrs, labels)
        
        running_loss += loss.item() * inputs.size(0)
        
        output_test(file, names, labels.cpu().detach().numpy(), aggrs.cpu().detach().numpy(), outputs.cpu().detach().numpy())
        
    loss = running_loss / len(dataloaders.dataset)
    file.close()
    
    return loss

def load_data(dir, mode="test"):
    name_data = []
    x_data = []
    y_data = []
    for folder in os.listdir(dir):
        for file in os.listdir(dir+folder):
            if file[-3:] == "png":
                # print(dir+folder+'/'+file)
                img = cv2.imread(dir+folder+'/'+file)
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                x_data.append(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
                if folder == "Closed":
                    name_data.append("Closed_"+file)
                    y_data.append([0])
                else:
                    name_data.append("Opened_"+file)
                    y_data.append([1])
            
    # y_data = preprocessing.minmax_scale(y_data,feature_range=(0,1))
    print(mode+"-Data loaded: "+str(len(name_data))+" images")
    return name_data, x_data, y_data
    
def transform_dataset(dataset, data_transforms):
    for i in range(len(dataset)):
        dataset[i] = data_transforms(dataset[i])
    return dataset
    
class myDataset(torch.utils.data.Dataset):
    def __init__(self, dataSource, mode="test"):
        # Just normalization for validation
        data_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        names, xs, ys = load_data(dataSource, mode)
        self.names = names
        self.imgs = transform_dataset(xs, data_transforms)
        self.labels = Variable(torch.FloatTensor(ys))

    def __getitem__(self, index):
        return self.names[index], self.imgs[index], self.labels[index]
        
    def __len__(self):
        return len(self.imgs)

# Detect if we have a GPU available
device = torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")

# Setup the loss fxn
criterion = nn.MSELoss()

model_ft = torch.load(model_dir, map_location={'cuda:1':'cuda:0'})
if isinstance(model_ft,torch.nn.DataParallel):
		model_ft = model_ft.module
model_ft.eval()

# Build testset
# Setup train test split
testsets = myDataset(data_dir)
testloader_dict = Data.DataLoader(testsets, batch_size=batch_size, shuffle=True, num_workers=4)
test_loss = test_model(model_ft, testloader_dict, criterion)
print("test mse: ", test_loss)