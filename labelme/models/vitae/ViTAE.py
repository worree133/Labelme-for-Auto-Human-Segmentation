"""
Rethinking Portrait Matting with Privacy Preserving
Inferernce file.

Copyright (c) 2022, Sihan Ma (sima7436@uni.sydney.edu.au) and Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting.git
Paper link: https://arxiv.org/abs/2203.16828

"""

import torch
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageQt
from skimage.transform import resize
from torchvision import transforms
import io
from labelme.models.vitae.config import *
from labelme.models.vitae.util import *
from labelme.models.vitae.network import build_model
from pathlib import Path
path = Path(__file__).parents[1]

def inference_once(model, scale_img, scale_trimap=None):
    if torch.cuda.device_count() > 0:
        tensor_img = torch.from_numpy(scale_img.astype(np.float32)[:, :, :]).permute(2, 0, 1).cuda()
    else:
        tensor_img = torch.from_numpy(scale_img.astype(np.float32)[:, :, :]).permute(2, 0, 1)
    input_t = tensor_img
    input_t = input_t/255.0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    input_t = normalize(input_t)
    input_t = input_t.unsqueeze(0)
    pred_global, pred_local, pred_fusion = model(input_t)[:3]
    pred_global = pred_global.data.cpu().numpy()
    pred_global = gen_trimap_from_segmap_e2e(pred_global)
    pred_local = pred_local.data.cpu().numpy()[0,0,:,:]
    pred_fusion = pred_fusion.data.cpu().numpy()[0,0,:,:]
    return pred_global, pred_local, pred_fusion

def inference_img_p3m(model, img):
    h, w, c = img.shape
    new_h = min(MAX_SIZE_H, h - (h % 32))
    new_w = min(MAX_SIZE_W, w - (w % 32))


    resize_h = int(h/2)
    resize_w = int(w/2)
    new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
    new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
    scale_img = resize(img,(new_h,new_w))*255.0
    pred_global, pred_local, pred_fusion = inference_once(model, scale_img)
    pred_local = resize(pred_local,(h,w))
    pred_global = resize(pred_global,(h,w))*255.0
    pred_fusion = resize(pred_fusion,(h,w))
    return pred_fusion


def loadVitae(img):

    model_path = os.path.join(path,'pretrained_models/p3mnet_pretrained_on_p3m10k.pth')
    img = ImageQt.fromqpixmap(img)
    img = np.array(img)[:, :, :3]
    ### build model
    model = build_model('vitae', pretrained=False)

    ### load ckpt
    if torch.cuda.is_available() is False:
        ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt['state_dict'], strict=True)

    else:
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'], strict=True)
        model = model.cuda()

    ### Test

    model.eval()

    h, w, c = img.shape
    if min(h, w) > SHORTER_PATH_LIMITATION:
        if h >= w:
            new_w = SHORTER_PATH_LIMITATION
            new_h = int(SHORTER_PATH_LIMITATION * h / w)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            new_h = SHORTER_PATH_LIMITATION
            new_w = int(SHORTER_PATH_LIMITATION * w / h)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    with torch.no_grad():
        if torch.cuda.device_count() > 0:
            torch.cuda.empty_cache()
        predict = inference_img_p3m(model, img)

    predict = predict * 255.0
    predict = cv2.resize(predict, (w, h), interpolation=cv2.INTER_LINEAR)
    #predict = cv2.cvtColor(predict, cv2.COLOR_BGR2RGB)
    #predict=Image.fromarray(np.uint8(predict))
    return predict.astype(np.uint8)
    #return predict.toqpixmap()
