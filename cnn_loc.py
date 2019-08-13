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
from seg_dict_save import save_dict
from model import *
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
command = "test"

# Dataset settings
num_images = 97200
sample_iter = 30
test_ratio = 0.1

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"
part_name = "fl"

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

# Dir settings
data_dir = 'datasets/shapenet_car_data/'
train_seg_dir = 'datasets/shapenet_car_seg/'
test_dir = 'datasets/shapenet_test_{}/'.format(part_name)
test_seg_dir = 'datasets/shapenet_test_{}_seg/'.format(part_name)
model_dir = 'params/sigmoid/{}_ft_{}.pkl'.format(model_name, part_name)
plot_dir = 'plots/sigmoid/{}_ft_{}.jpg'.format(model_name, part_name)
output_dir = 'outputs/sigmoid/{}_ft_{}.txt'.format(model_name, part_name)
html_dir = "htmls/sigmoid/{}_ft_{}.txt".format(model_name, part_name)


print("-------------------------------------")
print("Config:\nmodel:{}\nnum_classes:{}\nbatch size:{}\nepochs:{}\nsample set:{}\ntest set:{}".format(model_name, num_classes+2, batch_size, num_epochs, data_dir, test_dir))
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
        content += "]\n"
        file.write(content)
        html.write("{} gt:{} pred:{}\n".format(names[i], fl_gt, str(int(round(result[i][0]*data_range)))))

    
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
        dist = mean_absolute_error(outputs.cpu().detach().numpy()[0], labels.cpu().detach().numpy()[0])
        
        running_loss += loss.item() * inputs.size(0)
        running_dist += dist.item() * inputs.size(0)
        
        output_test(file, html, names, outputs.cpu().detach().numpy()[0])
        
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
                        dist = mean_absolute_error(outputs.cpu().detach().numpy()[0], labels.cpu().detach().numpy()[0])

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


def get_locat(mask):
    left = mask.shape[1]
    right = 0
    top = None
    bottom = None
    for i in range(mask.shape[0]):
        search = np.argwhere(mask[i]==1)
        if len(search)!=0 and top==None:
            top = i
        if len(search)==0 and top!=None and bottom==None:
            bottom = i-1
        if len(search)!=0 and top!=None and i==mask.shape[0]-1 and bottom==None:
            bottom = i
        if len(search)!=0:
            left = min(left, search[0][0])
            right = max(right, search[-1][0])
            
    print(left, right, top, bottom)
    if top!=None and bottom!=None:
        return (left+right)//2, (top+bottom)//2
    else:
        return None, None

def read_seg_dict(path):
    if not os.path.isfile(path):
        save_dict(part_name, seg_dir, path)
    seg_mask_dict = np.load(path).item()

    return seg_mask_dict

def load_data(dir, seg_dir, mode):
    seg_mask_dict = read_seg_dict(seg_dir)
    name_data = []
    mask_data = []
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
                        x, y = get_locat(seg_mask_dict[file])
                        if x == None or y == None:
                            continue
                        name_data.append(file)
                        img = cv2.imread(dir+file)
                        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                        x_data.append(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
                        y_data.append([int(fl)/data_range, x, y])
    else:
        num_test_images = int(num_images*test_ratio)
        random_list = range(num_images)
        test_id = random.sample(random_list, num_test_images)
        n = 0
        for file in os.listdir(dir):
                if file[-3:] == "png":
                    # if n in test_id:
                    type, fl, fr, bl, br, trunk, az, el, dist = file.split('_')
                    x, y = get_locat(seg_mask_dict[file])
                    if x == None or y == None:
                            continue
                    name_data.append(file)
                    img = cv2.imread(dir+file)
                    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                    x_data.append(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
                    y_data.append([int(fl)/data_range, x, y])
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
    def __init__(self, dataSource, segSource, mode):
        # Just normalization for validation
        data_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        names, xs, ys = load_data(dataSource, segSource, mode)
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
    model_ft, input_size = initialize_model(model_name, num_classes+2, feature_extract, use_pretrained=True)
    model_ft = nn.DataParallel(model_ft)


    # Print the model we just instantiated
    # print(model_ft)

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    trainsets = myDataset(data_dir, train_seg_dir, 'train')
    testsets = myDataset(test_dir, test_seg_dir, 'test')

    image_datasets = {'train': trainsets, 'val': testsets}


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
    model_ft, input_size = initialize_model(model_name, num_classes+2, feature_extract, use_pretrained=False)
    model_ft.load_state_dict(torch.load(model_dir))
    model_ft = nn.DataParallel(model_ft)
    if isinstance(model_ft,torch.nn.DataParallel):
            model_ft = model_ft.module
    model_ft.to(device)

    model_ft.eval()

    testsets = myDataset(test_dir, test_seg_dir, 'test')

# Build testset
testloader_dict = Data.DataLoader(testsets, batch_size=batch_size, shuffle=True, num_workers=4)
test_loss, test_dist = test_model(model_ft, testloader_dict, criterion)
print("test mse: ", test_loss)
print("test mae: ", test_dist*abs(data_range))