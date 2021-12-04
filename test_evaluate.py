#Evaluation of test generated images compared to ground truth image
#Input

import argparse
import logging
import os
from pathlib import Path

import torch
from tqdm import tqdm
import pickle

from modules.model import scale_img
from modules.evaluations.fid import calculate_fid
from pytorch_msssim import SSIM, MS_SSIM

from os.path import splitext
import numpy as np
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_options', nargs="+", default={'ssim','msssim','fid'})
    parser.add_argument('--img_scale', type=float, default=1.)

    # path args
    parser.add_argument('--run_name', type=str, default='', help='exp name')
    parser.add_argument('--data_root', type=str, default='/data/mohaa/project1/CelebAHQ')
    parser.add_argument('--gt_img_path', type=str, default='images')
    parser.add_argument('--test_root', type=str, default='/data/mohaa/project1/facial_mask_identity/test_result')
    parser.add_argument('--folders', nargs="+", default={})

    # additional args
    parser.add_argument('--specific_img', nargs="+", default={})

    args = parser.parse_args()

    #preprocess gt_img_path
    args.gt_img_path = os.path.join(args.data_root, args.gt_img_path)

    return args

def preprocess(pil_img, scale, is_mask=False):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    pil_img = pil_img.resize((newW, newH),
                                resample=Image.NEAREST if is_mask else Image.BICUBIC)
    img_ndarray = np.asarray(pil_img)

    if img_ndarray.ndim == 2 and not is_mask:
        img_ndarray = img_ndarray[np.newaxis, ...]
    if not is_mask:
        img_ndarray = img_ndarray.transpose((2, 0, 1))
        img_ndarray = img_ndarray / 255
        img_ndarray = torch.as_tensor(img_ndarray.copy()).float().contiguous()
    else:
        img_ndarray = torch.as_tensor(img_ndarray.copy()).long().contiguous()

    return img_ndarray

def load(filename):
    ext = splitext(filename)[1]
    if ext in ['.npz', '.npy']:
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)

def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Load test files id
    with open(Path(args.data_root)/Path('test_files_list.pkl'), 'rb') as f:
        testIds = pickle.load(f)

    if args.specific_img is not {}:
        testIds=list(args.specific_img)
    
    if 'ssim' in args.options:
        ssim_func = SSIM(data_range=1, size_average=True, channel=3)
    if 'ms_ssim' in args.options:
        ms_ssim_func = MS_SSIM(data_range=1, size_average=True, channel=3)
    
    result = []
    print ('Eval Metric Order: ',args.options)
    with tqdm(total=(len(testIds)*len(args.folders)), desc=f'Calculate evaluation', unit='img') as pbar:
        for id in testIds:
            gt_img_file = args.gt_img_path / Path(id + '.jpg')
            gt_img = load(gt_img_file)
            gt_img = preprocess(gt_img,0.25,False).to(device)
            for folder in args.folders:
                temp = []
                temp.append(id)
                temp.append(folder)
                gen_img_file = args.test_root / Path(folder) / Path(id + '_gen' + '.jpg')
                gen_img = load(gen_img_file)
                gen_img = preprocess(gen_img,1,False).to(device)
                for option in args.options:
                    if (option=='ssim'):
                        ssim = ssim_func(gt_img, gen_img)
                        temp.append(ssim)
                    if (option=='ms_ssim'):
                        ms_ssim = ms_ssim_func(gt_img,gen_img)
                        temp.append(ms_ssim)
                    if (option=='fid'):
                        fid_distance = calculate_fid(scale_img(gt_img, (299, 299)),
                                            scale_img(gen_img, (299, 299)), False,
                                            1)
                        temp.append(fid_distance)
                result.append(temp)
                pbar.update(1)
    with open(Path(args.test_root)/Path('test_evaluation.pkl'), 'wb') as f:
        pickle.dump(result, f)
    print ('test evaluation done!')

if __name__ == '__main__':
    main()
