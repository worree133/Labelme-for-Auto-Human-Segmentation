import os
import cv2
import toml

import numpy as np
from PIL import Image, ImageQt
import torch
from torch.nn import functional as F

import labelme.models.matteformer.utils as utils
from labelme.models.matteformer.utils import CONFIG
import labelme.models.matteformer.networks as networks
from pathlib import Path
path = Path(__file__).parents[1]


def single_inference(model, image_dict, checkcuda):

    with torch.no_grad(): 
        image, trimap = image_dict['image'], image_dict['trimap']
        if checkcuda is True:
            image = image.cuda()
            trimap = trimap.cuda()

        # run model
        pred = model(image, trimap)
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']

        # refinement
        alpha_pred = alpha_pred_os8.clone().detach()
        weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred, checkcuda, rand_width=CONFIG.model.self_refine_width1, train_mode=False)
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4>0]
        weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred, checkcuda, rand_width=CONFIG.model.self_refine_width2, train_mode=False)
        alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1>0]

        h, w = image_dict['alpha_shape']
        alpha_pred = alpha_pred[0, 0, ...].data.cpu().numpy() * 255
        alpha_pred = alpha_pred.astype(np.uint8)

        alpha_pred[np.argmax(trimap.cpu().numpy()[0], axis=0) == 0] = 0.0
        alpha_pred[np.argmax(trimap.cpu().numpy()[0], axis=0) == 2] = 255.

        alpha_pred = alpha_pred[32:h+32, 32:w+32]

        return alpha_pred


def generator_tensor_dict(image_path, trimap):
    # read images
    image = cv2.imread(image_path)
    trimap = ImageQt.fromqpixmap(trimap)
    trimap = cv2.cvtColor(np.array(trimap)[:, :, :3], cv2.COLOR_RGB2GRAY)

    sample = {'image': image, 'trimap':trimap, 'alpha_shape':(image.shape[0], image.shape[1])}

    # reshape
    h, w = sample["alpha_shape"]
    
    if h % 32 == 0 and w % 32 == 0:
        padded_image = np.pad(sample['image'], ((32,32), (32, 32), (0,0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((32,32), (32, 32)), mode="reflect")

        sample['image'] = padded_image
        sample['trimap'] = padded_trimap

    else:
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w
        padded_image = np.pad(sample['image'], ((32,pad_h+32), (32, pad_w+32), (0,0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((32,pad_h+32), (32, pad_w+32)), mode="reflect")

        sample['image'] = padded_image
        sample['trimap'] = padded_trimap

    # ImageNet mean & std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    # convert GBR images to RGB
    image, trimap = sample['image'][:,:,::-1], sample['trimap']

    # swap color axis
    image = image.transpose((2, 0, 1)).astype(np.float32)

    # trimap configuration
    padded_trimap[padded_trimap < 85] = 0
    padded_trimap[padded_trimap >= 170] = 2
    padded_trimap[padded_trimap >= 85] = 1

    # normalize image
    image /= 255.

    # to tensor
    sample['image'], sample['trimap'] = torch.from_numpy(image), torch.from_numpy(trimap).to(torch.long)
    sample['image'] = sample['image'].sub_(mean).div_(std)

    # trimap to one-hot 3 channel
    sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2, 0, 1).float()

    # add first channel
    sample['image'], sample['trimap'] = sample['image'][None, ...], sample['trimap'][None, ...]

    return sample




    # Parse configuration
def MatteFormer(image_path, trimap):

    with open(os.path.join(path,'matteformer/config/MatteFormer_Composition1k.toml')) as f:
        utils.load_config(toml.load(f))

    # Check if toml config file is loaded
    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.")


    # build model
    model = networks.get_generator(is_train=False)

    checkcuda = torch.cuda.is_available()
    if checkcuda is True:
        model.cuda()
        checkpoint = torch.load(os.path.join(path, 'pretrained_models/Matteformer.pth'))
    else:
        checkpoint = torch.load(os.path.join(path, 'pretrained_models/Matteformer.pth'), map_location=torch.device('cpu'))
    # load checkpoint

    model.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)

    # inference
    model = model.eval()


    image_dict = generator_tensor_dict(image_path, trimap)
    alpha_pred = single_inference(model, image_dict, checkcuda)

    # save images

    _al = cv2.cvtColor(alpha_pred, cv2.COLOR_GRAY2RGB)
    h, w, c = _al.shape

    cv2.imwrite('final.jpg', _al)
    return _al.astype(np.uint8)