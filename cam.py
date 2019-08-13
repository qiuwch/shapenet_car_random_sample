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
from model import *


############### Configurations ###############
# mode settings
generate_cam = True
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

# Model settings 
param_dir = "params/sigmoid/{}_ft_{}.pkl".format(model_name, part_name)
pred_dir = 'htmls/sigmoid/{}_ft_{}_same.txt'.format(model_name, part_name)

# Save settings
cam_dir = "cam_test/sigmoid/{}_ft_{}_same/".format(model_name, part_name)
over_save_dir = 'overlaps/sigmoid/{}_ft_{}_same.csv'.format(model_name, part_name)
focus_dir = 'focus_names/{}_ft_{}_same/focus.txt'.format(model_name, part_name)
unfocus_dir = 'focus_names/{}_ft_{}_same/unfocus.txt'.format(model_name, part_name)
none_dir = 'focus_names/{}_ft_{}_same/none.txt'.format(model_name, part_name)

# Division threshold
thresh = 0.2

# # Test Configurations
# test_dir = "datasets/test/"
# seg_dir = 'datasets/test_seg/'
# seg_dict_dir = 'seg_dict/test_seg.npy'

# param_dir = "params/sigmoid/resnet_ft_fl.pkl"
# pred_dir = 'htmls/test.txt'

# cam_dir = "cam_test/test/"
# over_save_dir = 'overlaps/test.csv'
# focus_dir = 'focus_names/test/focus.txt'
# unfocus_dir = 'focus_names/test/unfocus.txt'
# none_dir = 'focus_names/test/none.txt'


def read_seg_dict(path):
    if not os.path.isfile(path):
        save_dict(part_name, seg_dir, path)
    seg_mask_dict = np.load(path).item()

    return seg_mask_dict

def load_pred(path):
    file = open(path, 'r')
    content_list = file.readlines()
    ndict = {}
    for line in tqdm(content_list):
        content = line.strip().split(' ')
        ndict[content[0]] = [int(content[1].split(':')[-1]), int(content[2].split(':')[-1])]
    
    return ndict

def cal_ovlp(mask, heat):
    # Normalization for the heatmap
    data_transforms = transforms.Compose([
                transforms.ToTensor()
            ])
    heat = np.array(data_transforms(heat))
    result = np.sum(heat*mask)
    base = np.sum(mask)

    return np.sum(result)/base if base!= 0 else None

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
            img = cv2.imread(test_dir+file)
            img_pil = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

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




