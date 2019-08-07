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

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
#data_dir = "./data/hymenoptera_data"

# Dataset settings
num_images = 80000
train_ratio = 0.7
data_dir = 'shapenet_car_data/'

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 7

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Number of epochs to train for
num_epochs = 25

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

x_list = []
train_list = []
val_list = []

def output_test(file, names, outputs):
    for i in range(len(names)):
        # visualize txt
        type_gt, fl_gt, fr_gt, bl_gt, br_gt, trunk_gt, az_gt, el_gt, dist_gt = names[i].split('_')
        content  = "gt: [{}, {}, {}, {}, {}, {}, {}]  predictions: [{}, {}, {}, {}, {}, {}, {}]".format(fl_gt, fr_gt, bl_gt, br_gt, trunk_gt, az_gt, el_gt, \
        str(int(round(outputs[i][0]*-40))), str(int(round(outputs[i][1]*40))), str(int(round(outputs[i][2]*-40))), str(int(round(outputs[i][3]*40))), \
        str(int(round(outputs[i][4]*40))), str(int(round(outputs[i][5]*360))), str(int(round(outputs[i][6]*60+20))))
        content += "]\n"
        file.write(content)
        

def test_model(model, dataloaders, criterion):
    since = time.time()
    file = open("./test_out.txt",'w')
    
    running_loss = 0
    for names, inputs, labels in dataloaders:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        model.eval()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        
        running_loss += loss.item() * inputs.size(0)
        
        output_test(file, names, outputs.cpu().detach().numpy())
        
    loss = running_loss / len(dataloaders.dataset)
    file.close()
    
    return loss
        
    

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000.0


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        x_list.append(epoch)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (name, inputs, labels) in enumerate(dataloaders[phase]):
                #if epoch == 0:
                #    print(name)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                #running_corrects += torch.sum(preds == labels.data.long())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            #epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            
            # plot
            if phase == 'train':
                train_list.append(epoch_loss)
            else:
                val_list.append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_loss)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_model_wts
    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG16_bn
        """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size
    
def load_data(dir, mode, train_id):
    name_data = []
    x_data = []
    y_data = []
    n = 0
    num = 0
    for file in os.listdir(dir):
        if file[-3:] == "png":
            if mode == 'train' and n in train_id:
                name_data.append(file)
                img = cv2.imread(dir+file)
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                x_data.append(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
                type, fl, fr, bl, br, trunk, az, el, dist = file.split('_')
                y_data.append([int(fl), int(fr), int(bl), int(br), int(trunk), int(az), int(el)])
            elif mode == 'test' and n not in train_id and num < num_images*(1-train_ratio):
                name_data.append(file)
                img = cv2.imread(dir+file)
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                x_data.append(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
                type, fl, fr, bl, br, trunk, az, el, dist = file.split('_')
                y_data.append([int(fl), int(fr), int(bl), int(br), int(trunk), int(az), int(el)])
                num += 1
            n += 1
            
    y_data = preprocessing.minmax_scale(y_data,feature_range=(0,1))
    print(mode+"-Data loaded: "+str(len(name_data))+" images")
    return name_data, x_data, y_data
    
def transform_dataset(dataset, data_transforms):
    for i in range(len(dataset)):
        dataset[i] = data_transforms(dataset[i])
    return dataset
    
class myDataset(torch.utils.data.Dataset):
    def __init__(self, dataSource, mode, train_id):
        # Just normalization for validation
        data_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        names, xs, ys = load_data(dataSource, mode, train_id)
        self.names = names
        self.imgs = transform_dataset(xs, data_transforms)
        self.labels = Variable(torch.FloatTensor(ys))

    def __getitem__(self, index):
        return self.names[index], self.imgs[index], self.labels[index]
        
    def __len__(self):
        return len(self.imgs)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setup the loss fxn
criterion = nn.MSELoss()

# Setup train test split
num_train_images = int(num_images*train_ratio)
random_list = range(num_images)
train_id = random.sample(random_list, num_train_images)

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# keep training
# model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
# model_ft.load_state_dict(torch.load('./params.pkl'))

model_ft = nn.DataParallel(model_ft)

# Print the model we just instantiated
print(model_ft)

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
trainsets = myDataset(data_dir, 'train', train_id)
valsets = myDataset(data_dir, 'test', train_id)

image_datasets = {'train': trainsets, 'val': valsets}


# Create training and validation dataloaders
dataloaders_dict = {x: Data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Train and evaluate
model_ft, best_model_wts = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
torch.save(model_ft.module.state_dict(), 'params.pkl')


# plot
# plt.title('vgg16_bn Feature Extract',fontsize='large',fontweight='bold')
plt.title('vgg16_bn Fine-tune',fontsize='large', fontweight='bold')
#plt.title('ResNet18 Feature Extract',fontsize='large', fontweight='bold')
#plt.title('ResNet18 Fine-tune',fontsize='large', fontweight='bold')
#plt.title('DenseNet121 Feature Extract',fontsize='large',fontweight='bold')
#plt.title('DenseNet121 Fine-tuning',fontsize='large',fontweight='bold')
plt.plot(x_list,train_list,"x-",label="train loss")
plt.plot(x_list,val_list,"+-",label="val loss")
plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
plt.savefig("./loss_plot.jpg")

# Test
## Load model
# model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
# model_ft.load_state_dict(torch.load('./params.pkl'))
# if isinstance(model_ft,torch.nn.DataParallel):
# 		model_ft = model_ft.module
# model_ft.eval()

# Build testset
# testsets = myDataset(data_dir, 'test', train_id)
testloader_dict = Data.DataLoader(valsets, batch_size=batch_size, shuffle=True, num_workers=4)
test_loss = test_model(model_ft, testloader_dict, criterion)
print("test mse: ", test_loss)