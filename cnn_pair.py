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
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
#data_dir = "./data/hymenoptera_data"

# Train/Test mode
command = "train"

# Dataset settings
num_images = 97200
sample_iter = 30
test_ratio = 0.1
data_dir = 'datasets/shapenet_car_data/'
test_dir = 'datasets/shapenet_test_fl/'
model_dir = 'params/sigmoid/vgg_ft_fl.pkl'
plot_dir = 'plots/sigmoid/vgg_ft_fl.jpg'
output_dir = 'outputs/sigmoid/vgg_ft_fl.txt'
html_dir = "htmls/sigmoid/vgg_ft_fl.txt"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "vgg"

# Number of classes in the dataset
num_classes = 1

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Number of epochs to train for
num_epochs = 20

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

# Data range
data_range = -60

print("\n-------------------------------------")
print("Config:\nmodel:{}\nnum_classes:{}\nbatch size:{}\nepochs:{}\nsample set:{}\ntest set:{}".format(model_name, num_classes, batch_size, num_epochs, data_dir, test_dir))
print("-------------------------------------\n")

x_list = []
train_list = []
val_list = []

def output_test(file, html, names, result):
    for i in range(len(names)):
        # visualize txt
        type_gt, fl_gt, fr_gt, bl_gt, br_gt, trunk_gt, az_gt, el_gt, dist_gt = names[i].split('_')
        # content  = "gt: [ {} {} {} {} {} {} {}]---predictitmutmuxons: [".format(fl_gt, fr_gt, bl_gt, br_gt, trunk_gt, az_gt, el_gt)
        content  = "name: {}---gt: [ {}]---predictions: [".format(names[i], fl_gt)
        content += ' '+str(int(round(result[i][0]*data_range)))
        # content += ' '+str(int(round(result[i])))
        content += "]\n"
        file.write(content)
        html.write("{} gt:{} pred:{}\n".format(names[i], fl_gt, str(int(round(result[i][0]*data_range)))))
        # html.write("{} gt:{} pred:{}\n".format(names[i], fl_gt, str(int(round(result[i])))))

    
def test_model(model, dataloaders, criterion):
    since = time.time()
    file = open(output_dir,'w')
    html = open(html_dir,'w')
    
    running_loss = 0
    running_dist = 0
    for names, inputs, labels in dataloaders:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        model.eval()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        dist = mean_absolute_error(outputs.cpu().detach().numpy(), labels.cpu().detach().numpy())
        
        running_loss += loss.item() * inputs.size(0)
        running_dist += dist.item() * inputs.size(0)
        
        output_test(file, html, names, outputs.cpu().detach().numpy())
        # output_test(file, html, names, preprocessing.minmax_scale(outputs.cpu().detach().numpy()[:,0],feature_range=(-40,0)))
        
    loss = running_loss / len(dataloaders.dataset)
    dist = running_dist / len(dataloaders.dataset)
    file.close()
    html.close()
    
    return loss, dist
        
    

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        x_list.append(epoch)
        best_loss = 10000

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_dist = 0.0

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
                        dist = mean_absolute_error(outputs.cpu().detach().numpy(), labels.cpu().detach().numpy())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_dist += dist.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_dist = running_dist / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}, Dist: {:.4f}'.format(phase, epoch_loss, epoch_dist*abs(data_range)))
            
            # plot
            if phase == 'train':
                train_list.append(epoch_dist*60)
            else:
                val_list.append(epoch_dist*60)

            # deep copy the model
            if phase == 'val' and (epoch == 0 or epoch_loss<best_loss):
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

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
        model_ft.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Sigmoid() # add sigmoid or not
        )
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
        model_ft = models.vgg16_bn(pretrained=False)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
            nn.Sigmoid() # add sigmoid or not
        )
        pretrained_dict = load_state_dict_from_url(model_urls["vgg16_bn"],
                                              progress=True)
        model_dict =  model_ft.state_dict()
        state_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model_ft.load_state_dict(model_dict)
        model_ft.classifier[6] = nn.Linear(4096,512)
        model_ft.classifier[9] = nn.Linear(512,num_classes)
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

def sample_data():
    fl = [x for x in range(-40, 1, 20)]
    fr = [x for x in range(0, 60, 20)]
    bl = [x for x in range(-40, 1, 20)]
    br = [x for x in range(0, 60, 20)]
    trunk = [x for x in range(0, 60, 20)] 
    az = [x for x in range(0, 361, 40)]
    el = [x for x in range(20, 90, 20)]
    dist = [400, 450]
    fl_spl = random.sample(fl, 1)
    fr_spl = random.sample(fr, 1)
    bl_spl = random.sample(bl, 1)
    br_spl = random.sample(br, 1)
    trunk_spl = random.sample(trunk, 1)
    return str(fl_spl[0]), str(fr_spl[0]), str(bl_spl[0]), str(br_spl[0]), str(trunk_spl[0])

def load_data(dir, mode):
    name_data = []
    x_data = []
    y_data = []
    if mode == 'train':
        print("Start sampling...")
        for i in tqdm(range(sample_iter)):
            fl_spl, fr_spl, bl_spl, br_spl, trunk_spl = sample_data()
            for file in os.listdir(dir):
                if file[-3:] == "png":
                    type, fl, fr, bl, br, trunk, az, el, dist = file.split('_')
                    if bl == bl_spl and fr == fr_spl and br == br_spl and trunk == trunk_spl:
                        name_data.append(file)
                        img = cv2.imread(dir+file)
                        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                        x_data.append(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
                        y_data.append([int(fl)/data_range])
    else:
        num_test_images = int(num_images*test_ratio)
        random_list = range(num_images)
        test_id = random.sample(random_list, num_test_images)
        n = 0
        for file in os.listdir(dir):
                if file[-3:] == "png":
                    # if n in test_id:
                    type, fl, fr, bl, br, trunk, az, el, dist = file.split('_')
                    name_data.append(file)
                    img = cv2.imread(dir+file)
                    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                    x_data.append(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
                    y_data.append([int(fl)/data_range])
                    n += 1
                
    y_data = preprocessing.minmax_scale(y_data,feature_range=(0,1))
    # print(y_data)
    print(mode+"-Data loaded: "+str(len(name_data))+" images")
    return name_data, x_data, y_data
    
def transform_dataset(dataset, data_transforms):
    for i in range(len(dataset)):
        dataset[i] = data_transforms(dataset[i])
    return dataset
    
class myDataset(torch.utils.data.Dataset):
    def __init__(self, dataSource, mode):
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

if command == "train":
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    model_ft = nn.DataParallel(model_ft)


    # Print the model we just instantiated
    # print(model_ft)

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    trainsets = myDataset(data_dir, 'train')
    valsets = myDataset(test_dir, 'test')

    #image_datasets = {'train': myDataset([transform_dataset(X_train, data_transforms), Variable(torch.FloatTensor(y_train))]), 'val': myDataset([transform_dataset(X_val, data_transforms), Variable(torch.FloatTensor(y_val))])}
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
    model_ft = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
    torch.save(model_ft.module.state_dict(), model_dir)

    # plot
    # plt.title('vgg16_bn Feature Extract',fontsize='large',fontweight='bold')
    plt.title('vgg16_bn Fine-tune',fontsize='large', fontweight='bold')
    #plt.title('ResNet18 Feature Extract',fontsize='large', fontweight='bold')
    # plt.title('ResNet18 Fine-tune',fontsize='large', fontweight='bold')
    #plt.title('DenseNet121 Feature Extract',fontsize='large',fontweight='bold')
    #plt.title('DenseNet121 Fine-tuning',fontsize='large',fontweight='bold')
    plt.plot(x_list,train_list,"x-",label="train loss")
    plt.plot(x_list,val_list,"+-",label="val loss")
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.savefig(plot_dir)

# Test
# Load model
if command == "test":
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
    model_ft.load_state_dict(torch.load(model_dir))
    model_ft = nn.DataParallel(model_ft)
    if isinstance(model_ft,torch.nn.DataParallel):
            model_ft = model_ft.module
    model_ft.to(device)

    model_ft.eval()

    valsets = myDataset(test_dir, 'test')

# Build testset
testloader_dict = Data.DataLoader(valsets, batch_size=batch_size, shuffle=True, num_workers=4)
test_loss, test_dist = test_model(model_ft, testloader_dict, criterion)
print("test mse: ", test_loss)
print("test mae: ", test_dist*abs(data_range))