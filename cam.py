# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

import io
import os
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn as nn
import torch
import numpy as np
import cv2
import pdb
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import csv
import re
import math
from seg_dict_save import save_dict

# mode settings
generate_cam = False
cal_overlap = True

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"
part_name = 'fl'

# Number of classes in the dataset
num_classes = 1

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

# Dataset directory
test_dir = "datasets/shapenet_test_{}/".format(part_name)
seg_dir = 'datasets/shapenet_test_{}_seg/'.format(part_name)
seg_dict_dir = 'seg_dict/shapenet_test_{}_seg.npy'.format(part_name)

# overlap settings 
cam_dir = "cam_test/sigmoid/{}_ft_{}_same/".format(model_name, part_name)
param_dir = "params/sigmoid/{}_ft_{}.pkl".format(model_name, part_name)
pred_dir = 'htmls/sigmoid/{}_ft_{}_same.txt'.format(model_name, part_name)
over_save_dir = 'overlaps/sigmoid/{}_ft_{}_same.csv'.format(model_name, part_name)
focus_dir = 'focus_names/{}_ft_{}_same/focus.txt'.format(model_name, part_name)
unfocus_dir = 'focus_names/{}_ft_{}_same/unfocus.txt'.format(model_name, part_name)
none_dir = 'focus_names/{}_ft_{}_same/none.txt'.format(model_name, part_name)

thresh = 0.1

# Test Configurations
# test_dir = "datasets/test/"
# cam_dir = "cam_test/test/"
# param_dir = "params/resnet_ft_fl.pkl"

# part_name = 'fl'
# # if need to generate dict.npy
# seg_dir = 'datasets/test_seg/'
# # if dict.npy has generated
# seg_dict_dir = 'seg_dict/test_seg.npy'
# pred_dir = 'htmls/test.txt'
# over_save_dir = 'overlaps/test.csv'


def read_seg_dict(path):
    if not os.path.isfile(path):
        save_dict(seg_dir, path)
    seg_mask_dict = np.load(path).item()

    return seg_mask_dict

def cal_ovlp(mask, heat):
    data_transforms = transforms.Compose([
                transforms.ToTensor()
            ])
    heat = np.array(data_transforms(heat))
    result = np.sum(heat*mask)
    base = np.sum(mask)

    return np.sum(result)/base if base!= 0 else None

def load_pred(path):
    file = open(path, 'r')
    content_list = file.readlines()
    ndict = {}
    for line in tqdm(content_list):
        content = line.strip().split(' ')
        ndict[content[0]] = [int(content[1].split(':')[-1]), int(content[2].split(':')[-1])]

    
    return ndict



# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
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
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )
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


# original
# net = models.resnet18(pretrained=True)

# modified
net, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
net.load_state_dict(torch.load(param_dir))

if model_name == 'resnet':
    finalconv_name  = 'layer4'
elif model_name == 'vgg':
    finalconv_name  = 'features'

net.eval()

# traverse all the images in test_dir
print("Load predictions...")
pred_dict = load_pred(pred_dir)
print("Load seg masks...")
seg_mask_dict = read_seg_dict(seg_dict_dir)

# data init
focus_loss = 0
unfocus_loss = 0
none_loss = 0

focus_num = 0
unfocus_num = 0
none_num = 0

# save dir init
focus_file = open(focus_dir, 'w')
unfocus_file = open(unfocus_dir, 'w')
none_file = open(none_dir, 'w')

with open(over_save_dir,"w") as csvfile: 
    over_file = csv.writer(csvfile)
    print("Start CAM...")
    over_file.writerow(["filename","fl","fr","bl","br","trunk","az","el","dist","overlap","l1 angle error"])
    for file in tqdm(os.listdir(test_dir)):
        if file[-3:] == "png":
        # hook the feature extractor
            features_blobs = []
            def hook_feature(module, input, output):
                features_blobs.append(output.data.cpu().numpy())
            net._modules.get(finalconv_name).register_forward_hook(hook_feature)

            # get the softmax weight
            params = list(net.parameters())
            weight_softmax = np.squeeze(params[-2].data.numpy()).reshape(1,512)

            def returnCAM(feature_conv, weight_softmax, class_idx):
                # generate the class activation maps upsample to 256x256
                size_upsample = (256, 256)
                bz, nc, h, w = feature_conv.shape
                output_cam = []
                for idx in class_idx:
                    cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
                    cam = cam.reshape(h, w)
                    cam = cam - np.min(cam)
                    cam_img = cam / np.max(cam)
                    cam_img = np.uint8(255 * cam_img)
                    output_cam.append(cv2.resize(cam_img, size_upsample))
                return output_cam


            normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
            preprocess = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize
            ])

            #################################################################
            # print("{} is being tested...".format(file))
            img_pil = Image.open(test_dir+file).convert('RGB')

            img_tensor = preprocess(img_pil)
            img_variable = Variable(img_tensor.unsqueeze(0))
            logit = net(img_variable)

            h_x = F.softmax(logit, dim=1).data.squeeze()
            probs, idx = h_x.sort(0, True)
            probs = probs.numpy()
            idx = idx.numpy()

            # generate class activation mapping for the top1 prediction
            CAMs = returnCAM(features_blobs[0], weight_softmax, [idx])

            # render the CAM and output
            img = cv2.imread(test_dir+file)
            height, width, _ = img.shape
            heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)

            if generate_cam:
                result = heatmap * 0.3 + img * 0.5
                cv2.imwrite(cam_dir+file, result)

            if cal_overlap:
                score = cal_ovlp(seg_mask_dict[file], cv2.resize(CAMs[0],(width, height)))
                # print(score)
                type, fl, fr, bl, br, trunk, az, el, dist, _ = re.split(r'[_.]', file)
                loss = math.sqrt(mean_squared_error([pred_dict[file][0]], [pred_dict[file][1]]))
                if score != None and (score > thresh):# or score < 1-thresh):
                    focus_loss += loss
                    focus_num += 1
                    focus_file.write("{} {}\n".format(file, score))
                elif score != None and (score <= thresh):# and score >= 1-thresh):
                    unfocus_loss += loss
                    unfocus_num += 1
                    unfocus_file.write("{} {}\n".format(file, score))
                else:
                    none_loss += loss
                    none_num += 1
                    none_file.write("{} {}\n".format(file, score))

                over_file.writerow([file, fl, fr, bl, br, trunk, az, el, dist, str(score), str(loss)])

    focus_loss = focus_loss / focus_num
    unfocus_loss = unfocus_loss / unfocus_num
    none_loss = none_loss / none_num
    print('{} focus images, loss: {}'.format(focus_num, focus_loss))
    print('{} unfocus images, loss: {}'.format(unfocus_num, unfocus_loss))
    print('{} None images, loss: {}'.format(none_num, none_loss))

    focus_file.close()
    unfocus_file.close()
    none_file.close()




