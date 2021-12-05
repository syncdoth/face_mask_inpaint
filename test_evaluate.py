"""Evaluation of test generated images compared to ground truth image Input"""

import argparse
import os

import pandas as pd
import torch
from pytorch_msssim import MS_SSIM, SSIM
from tqdm import tqdm

from dataloader import BasicDataset
from modules.evaluations.fid import PartialInceptionNetwork, calculate_fid
from modules.model import scale_img


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_options', nargs="+", default={'ssim', 'ms_ssim', 'fid'})
    parser.add_argument('--batch_size', type=int, default=8)

    # path args
    parser.add_argument('--data_root', type=str, default='/data/mohaa/project1/CelebAHQ')
    parser.add_argument('--gt_img_path', type=str, default='images')
    parser.add_argument('--test_folder', type=str, default='')

    # additional args
    parser.add_argument('--specific_img', nargs="+", default={})

    args = parser.parse_args()

    #preprocess gt_img_path
    args.gt_img_path = os.path.join(args.data_root, args.gt_img_path)

    return args


def load_images(args, test_id):
    gt_img_file = os.path.join(args.gt_img_path, f'{test_id}.jpg')
    gt_img = BasicDataset.load(gt_img_file)
    gt_img = BasicDataset.preprocess(gt_img, 0.25, False)

    gen_img_file = os.path.join(args.test_folder, f'gen_{test_id}.jpg')
    gen_img = BasicDataset.load(gen_img_file)
    gen_img = BasicDataset.preprocess(gen_img, 1, False)
    return gt_img, gen_img


def make_batch(test_ids, batch_size):
    for i in range(0, len(test_ids), batch_size):
        yield test_ids[i:min(i + batch_size, len(test_ids))]


def main():
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Load test files id
    test_ids = [
        os.path.basename(x).split('.')[0].split('_')[1]
        for x in os.listdir(args.test_folder)
        if x.startswith('gen')
    ]

    if args.specific_img:
        test_ids = list(args.specific_img)

    if 'ssim' in args.eval_options:
        ssim_func = SSIM(data_range=1, size_average=True, channel=3)
    if 'ms_ssim' in args.eval_options:
        ms_ssim_func = MS_SSIM(data_range=1, size_average=True, channel=3)
    if 'fid' in args.eval_options:
        inception_network = PartialInceptionNetwork()
        inception_network = inception_network.to(device)
        inception_network.eval()

    print('Eval Metric Order: ', args.eval_options)

    eval_results = {k: 0 for k in args.eval_options}

    pbar = tqdm(make_batch(test_ids, args.batch_size), total=len(test_ids))
    for batch_ids in pbar:
        batch_imgs = [load_images(args, bid) for bid in batch_ids]
        gt_img = torch.stack([imgs[0] for imgs in batch_imgs]).to(device)
        gen_img = torch.stack([imgs[1] for imgs in batch_imgs]).to(device)

        if 'ssim' in args.eval_options:
            ssim = ssim_func(gt_img, gen_img)
            eval_results['ssim'] += float(ssim) * len(batch_ids)
        if 'ms_ssim' in args.eval_options:
            ms_ssim = ms_ssim_func(gt_img, gen_img)
            eval_results['ms_ssim'] += float(ms_ssim) * len(batch_ids)
        if 'fid' in args.eval_options:
            fid_distance = calculate_fid(scale_img(gt_img, (299, 299)),
                                         scale_img(gen_img, (299, 299)), len(batch_ids),
                                         inception_network)
            eval_results['fid'] += float(fid_distance) * len(batch_ids)

        pbar.update(len(batch_ids))

    df = pd.DataFrame({k: [v / len(test_ids)] for k, v in eval_results.items()})
    print(df)
    df.to_csv(os.path.join(args.test_folder, 'metrics.csv'), index=False)


if __name__ == '__main__':
    main()
