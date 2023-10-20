from __future__ import print_function
from itertools import starmap

import time

# from utils import list_images
import os
from turtle import st
import h5py
import numpy as np
import matplotlib.pyplot as plt
from train import train
from generate import generate
import argparse
from PIL import Image
from natsort import natsorted
from tqdm import tqdm
import statistics



BATCH_SIZE = 24
EPOCHES = 1
LOGGING = 40
MODEL_SAVE_PATH = './model/'
IS_TRAINING = False

def img2RGB(f_name, vi_name):
    vi_img = Image.open(vi_name)
    vi_img = vi_img.convert('YCbCr')
    vi_Y, vi_Cb, vi_Cr = vi_img.split()
    f_img = Image.open(f_name).convert('L')
    f_img = Image.merge('YCbCr', (f_img, vi_Cb, vi_Cr))
    f_RGB = f_img.convert('RGB')
    f_RGB.save(f_name)

def main(Method = 'FusionGAN', model_path='', ir_dir='', vi_dir='', save_dir='', is_RGB=True):  
   
    time_list = []
    pet_path = vi_dir
    mri_path = ir_dir    
    os.makedirs(save_dir, exist_ok=True)
    filelist = natsorted(os.listdir(pet_path))
    test_bar = tqdm(filelist)
    for i, item in enumerate(test_bar):
        if item.endswith('.bmp') or item.endswith('.png') or item.endswith('.jpg') or item.endswith('.tif'):
            pet_name = os.path.join(os.path.abspath(pet_path), item)
            mri_name = os.path.join(os.path.abspath(mri_path), item)
            fused_image_name = os.path.join(os.path.abspath(save_dir), item)
            test_time = generate(pet_name, mri_name, model_path, i, save_name=fused_image_name)
            time_list.append(test_time)        
            # if is_RGB:
            #     img2RGB(fused_image_name, pet_name)                
            test_bar.set_description('{} | {} | {:.4f} s'.format(Method, item, test_time))
                
    print('时间均值: {:.4f}, 标准差: {:.4f}'.format(statistics.mean(time_list), statistics.stdev(time_list)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Method', type=str, default='TarDAL', help='Method name')
    parser.add_argument('--model_path', type=str, default='./model/model.ckpt', help='pretrained weights path')
    parser.add_argument('--ir_dir', type=str, default='../datasets/PET-MRI/MRI', help='infrared images dir')
    parser.add_argument('--vi_dir', type=str, default='../datasets/PET-MRI/PET', help='visible image dir')
    parser.add_argument('--save_dir', type=str, default='../Results/PET-MRI/DDcGAN', help='fusion results dir')
    parser.add_argument('--is_RGB', type=bool, default=True, help='colorize fused images with visible color channels')
    opts = parser.parse_args()
    main(
        Method=opts.Method, 
        model_path=opts.model_path,
        ir_dir = opts.ir_dir,
        vi_dir = opts.vi_dir,
        save_dir = opts.save_dir,
        is_RGB=opts.is_RGB
    )

