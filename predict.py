import argparse
import logging
import torch
import random

import numpy as np
import torch.nn.functional as F

from scripts.model import Generator_main, Generator_branch
from scripts.dataset import LearningAVSegData
from scripts.eval import eval_net
from scripts.utils import Define_image_size

from torch.utils.data import DataLoader
from PIL import Image, ImageEnhance

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch-size', type=int, default=6, help='Batch size', dest='batchsize')
    parser.add_argument('--job_name', type=str, default='J', help='type of discriminator', dest='jn')
    parser.add_argument('--dataset', type=str, help='test dataset name', dest='dataset')
    # parser.add_argument('--uniform', type=str, default='False', help='whether to uniform the image size', dest='uniform')
    parser.add_argument('--input', '-i', type=str, required=True, help='input image file')
    parser.add_argument('--output', '-o', type=str, required=True, help='output image file')

    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    has_gpu = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_built()
    device = "mps" if has_mps else "gpu" if has_gpu else "cpu"

    logging.info(f'Using device {device}')

    img_size = (720,720)
    dataset_name = args.dataset

    input_image = Image.open(args.input)
    input_image = input_image.resize(img_size)
    # en = ImageEnhance.Color(input_image)
    # input_image = en.enhance(random.uniform(0.8,1.2))
    input_image = np.asarray(input_image).astype(np.float32)

    mean=np.mean(input_image[input_image[...,0] > 00.0],axis=0)
    std=np.std(input_image[input_image[...,0] > 00.0],axis=0)
    input_image=(input_image-1.0*mean)/1.0*std

    if len(input_image.shape) == 2:
        input_image = np.expand_dims(input_image, axis=2)

    input_image = np.transpose(input_image, (2, 0, 1))

    data = torch.from_numpy(input_image).type(torch.FloatTensor)
    data = data.unsqueeze(0)

    checkpoint_path_ = "./{}/{}_{}/Discriminator_unet/".format(dataset_name, args.jn, 42, dataset_name)

    net_G = Generator_main(input_channels=3, n_filters=32, n_classes=4, bilinear=False)
    net_G_A = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
    net_G_V = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)

    net_G.load_state_dict(torch.load(checkpoint_path_ + 'CP_best_F1_all.pth', map_location=torch.device(device)))
    net_G_A.load_state_dict(torch.load(checkpoint_path_ + 'CP_best_F1_A.pth', map_location=torch.device(device)))
    net_G_V.load_state_dict(torch.load(checkpoint_path_ + 'CP_best_F1_V.pth', map_location=torch.device(device)))
        
    net_G.eval()
    net_G_A.eval()
    net_G_V.eval()

    net_G.to(device=device)
    net_G_A.to(device=device)
    net_G_V.to(device=device)

    with torch.no_grad():
        data = data.to(device=device, dtype=torch.float32)
        # data = data.unsqueeze(0)
        
        _,masks_pred_G_A_fusion = net_G_A(data)
        _,masks_pred_G_V_fusion = net_G_V(data)

        masks_pred_G_A_part = masks_pred_G_A_fusion.detach()
        masks_pred_G_V_part = masks_pred_G_V_fusion.detach()
        mask_pred,_,_,_ = net_G(data, masks_pred_G_A_part, masks_pred_G_V_part)

        mask_pred_tensor_small = mask_pred.clone().detach()
        mask_pred_tensor_small = F.softmax(mask_pred_tensor_small,dim=1)
        mask_pred_tensor_small = torch.squeeze(mask_pred_tensor_small)

        _,prediction_decode = torch.max(mask_pred_tensor_small, 0)
        prediction_decode=prediction_decode.type(torch.FloatTensor)

        mask_pred = F.interpolate(mask_pred, (720, 720), mode='bilinear', align_corners=True)

    # mask_pred_softmax = mask_pred
    # mask_pred_softmax_cpu = mask_pred_softmax.detach().cpu()
    # _,mask_pred_softmax_cpu_decode = torch.max(F.softmax(mask_pred_softmax_cpu,dim=1),1)
    # mask_pred_softmax_cpu_decode=mask_pred_softmax_cpu_decode.numpy()
    # mask_pred_softmax_cpu_decode = np.squeeze(mask_pred_softmax_cpu_decode)
    # mask_pred_softmax_cpu_decode = mask_pred_softmax_cpu_decode.astype(np.uint8)

    # save output of the model to the output file
    # Image.fromarray(mask_pred_softmax_cpu_decode).save(args.output)
        
    # save masks_pred_G_A_part an image to the output file
    masks_pred_G_A_fusion = masks_pred_G_A_fusion.detach().cpu()
    masks_pred_G_A_fusion = torch.max(F.softmax(masks_pred_G_A_fusion,dim=1),1)
    masks_pred_G_A_fusion=masks_pred_G_A_fusion.values.numpy()
    masks_pred_G_A_fusion = np.squeeze(masks_pred_G_A_fusion)
    masks_pred_G_A_fusion = masks_pred_G_A_fusion.astype(np.uint8)
    Image.fromarray(masks_pred_G_A_fusion).save(args.output)

    logging.info(f'Inference output saved to {args.output}')
