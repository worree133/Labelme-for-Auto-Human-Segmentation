import os
import torch
import torchvision
import cv2
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from skimage import io, transform, color
from PIL import Image, ImageQt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os
from pathlib import Path

from .model import ControlU2Netpseg, U2NETP
# ------- 1. define loss function --------

path = Path(__file__).parents[1]

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def save_output(pred, img_shape):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    predict_np = predict_np*255
    predict_np = cv2.resize(predict_np, img_shape)

    return predict_np

    
def rescale(image, semantic, scale=320):

    h, w = image.shape[:2]

    if isinstance(scale, int):
        if h > w:
            new_h, new_w = scale * h / w, scale
        else:
            new_h, new_w = scale, scale * w / h
    else:
        new_h, new_w = scale

    new_h, new_w = int(new_h), int(new_w)

    # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
    # img = transform.resize(image,(new_h,new_w),mode='constant')
    # lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

    img = transform.resize(image, (scale, scale), mode='constant')
    sem = transform.resize(semantic, (scale, scale), mode='constant')


    return img, sem


def toTensor(image, sem):

    # change the color space

    tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
    tmpSem = np.zeros((sem.shape[0], sem.shape[1], 3))
    image = image / np.max(image)
    sem = sem / np.max(sem)
    if image.shape[2] == 1:
        tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
        tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        tmpSem[:, :, 0] = (sem[:, :, 0] - 0.485) / 0.229
        tmpSem[:, :, 1] = (sem[:, :, 0] - 0.485) / 0.229
        tmpSem[:, :, 2] = (sem[:, :, 0] - 0.485) / 0.229
    else:
        tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
        tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
        tmpSem[:, :, 0] = (sem[:, :, 0] - 0.485) / 0.229
        tmpSem[:, :, 1] = (sem[:, :, 1] - 0.456) / 0.224
        tmpSem[:, :, 2] = (sem[:, :, 2] - 0.406) / 0.225


    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpSem = tmpSem.transpose((2, 0, 1))


    return torch.from_numpy(tmpImg), torch.from_numpy(tmpSem)


def CtrlU2Net(img, sem):


    # --------- 1. get image path and name ---------
    img = ImageQt.fromqpixmap(img)
    img = np.array(img)[:, :, :3]
    img_shape = (img.shape[1], img.shape[0])
    #img = img.reshape((1,3,img.shape[0], img.shape[1]))
    sem = ImageQt.fromqpixmap(sem)
    sem = np.array(sem)[:, :, :3]


    img, sem = rescale(img,sem)
    img, sem = toTensor(img,sem)
    #sem = sem.reshape((1,3,sem.shape[0], sem.shape[1]))
    model_dir = os.path.join(path, 'pretrained_models/CtrlU2Netp.pth')

    net = ControlU2Netpseg()
    #net = U2NETP()
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    img = img[np.newaxis,:,:]
    sem = sem[np.newaxis, :, :]


    img = img.type(torch.FloatTensor)
    sem = sem.type(torch.FloatTensor)

    if torch.cuda.is_available():
        img = Variable(img.cuda())
        sem = Variable(sem.cuda())
    else:
        img = Variable(img)
        sem = Variable(sem)

    d1,d2,d3,d4,d5,d6,d7= net(img, sem)

    # normalization
    pred = d1[:,0,:,:]
    pred = normPRED(pred)

    pred = save_output(pred,img_shape )

    del d1,d2,d3,d4,d5,d6,d7

    return pred

