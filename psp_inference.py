import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_msssim import MS_SSIM, SSIM
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import ReferenceDataset
# from modules.evaluations.fid import calculate_fid, PartialInceptionNetwork
from modules.mask_detector import MaskDetector
from modules.pluralistic_model import base_function
# psp
from modules.psp.psp import pSp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/data/mohaa/project1/CelebAHQ')
    parser.add_argument('--identity_file_path',
                        type=str,
                        default='CelebA-HQ-identity.txt')
    parser.add_argument('--mask_path', type=str, default='binary_map')
    parser.add_argument('--src_img_path', type=str, default='images_masked_test')
    parser.add_argument('--ref_img_path', type=str, default='images')
    parser.add_argument('--mask_detector_path',
                        type=str,
                        default='saved_model/new_mask_detector.pth')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--pt_ckpt_path',
                        default='pretrained_models/psp_ffhq_encode.pt',
                        type=str,
                        help='Path to pretrained pSp model checkpoint')

    # pSp args: DO NOT MODIFY
    parser.add_argument('--use_ref', action='store_true', help='use reference image')
    parser.add_argument('--use_attention', default=0, type=int, help='use attention')
    parser.add_argument('--encoder_type', type=str, default='GradualStyleEncoder')
    parser.add_argument('--output_size',
                        default=1024,
                        type=int,
                        help='Output size of generator')
    parser.add_argument('--train_decoder',
                        default=0,
                        type=int,
                        help='Whether to train the decoder model')
    parser.add_argument(
        '--start_from_latent_avg',
        type=int,
        default=1,
        help='Whether to add average latent vector to generate codes from encoder.')
    parser.add_argument('--learn_in_w',
                        type=int,
                        default=0,
                        help='Whether to learn in w space instead of w+')
    parser.add_argument('--randomize_noise',
                        type=int,
                        default=0,
                        help='whether to randomize noise in stylegan')
    # pretrained weight paths
    parser.add_argument('--stylegan_weights',
                        default=None,
                        type=str,
                        help='Path to StyleGAN model weights')

    args = parser.parse_args()

    # process data path args here
    args.src_img_path = os.path.join(args.data_root, args.src_img_path)
    args.ref_img_path = os.path.join(args.data_root, args.ref_img_path)
    args.mask_path = os.path.join(args.data_root, args.mask_path)
    args.identity_file_path = os.path.join(args.data_root, args.identity_file_path)

    return args


@torch.no_grad()
def infer_batch(generator, mask_detector, batch_images, device):
    generator.eval()

    if len(batch_images) == 1:
        src_img = batch_images[0]
        src_img = src_img.to(device)
        ref_img = src_mask = None
    else:
        src_img, ref_img = batch_images
        src_img = src_img.to(device)
        ref_img = ref_img.to(device)
        src_mask = mask_detector((src_img + 1) / 2, mode='train')  # [N, 2, H, W]
        src_mask = src_mask.argmax(1).float()  # [N, H, W]

    gen_images, _ = generator(src_img,
                              ref=ref_img,
                              src_mask=src_mask,
                              return_latents=True,
                              resize=True,
                              randomize_noise=False)  # [N, 3, H, W]

    src_mask = src_mask.detach().cpu() if src_mask is not None else None
    return gen_images.detach(), src_mask


def tensor2im(var):
    var = var.permute(1, 2, 0).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def evaluate(gt_img, gen_img, ssim_func, ms_ssim_func):
    ssim = ssim_func(gt_img, ((gen_img + 1) / 2))
    ms_ssim = ms_ssim_func(gt_img, ((gen_img + 1) / 2))
    # fid_distance = calculate_fid(scale_img(gt_img, (299, 299)),
    #                              scale_img(gen_img, (299, 299)), 2, inception_network)

    return float(ssim), float(ms_ssim)  #, fid_distance


def main():
    args = get_args()

    ssim_func = SSIM(data_range=1, size_average=True, channel=3)
    ms_ssim_func = MS_SSIM(data_range=1, size_average=True, channel=3)
    # inception_network = PartialInceptionNetwork()

    # MODEL SETUP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # inception_network = inception_network.to(device)
    # inception_network.eval()

    # load saved mask detector
    mask_detector = MaskDetector(n_channels=3, bilinear=True)
    if args.mask_detector_path:
        mask_detector.load_state_dict(torch.load(args.mask_detector_path))
    base_function._freeze(mask_detector)  # freeze
    mask_detector = mask_detector.to(device)
    # define models
    generator = pSp(args).to(device)
    if generator.latent_avg is None:
        generator.latent_avg = generator.decoder.mean_latent(int(1e5))[0].detach()

    mask_detector.eval()
    generator.eval()

    # DATA SETUP
    dataset = ReferenceDataset(args.src_img_path,
                               args.ref_img_path,
                               args.mask_path,
                               args.identity_file_path,
                               apply_transform=True,
                               scale=0.25,
                               use_ssim=True,
                               device=device,
                               return_id=True)
    test_loader = DataLoader(dataset,
                             shuffle=False,
                             drop_last=False,
                             batch_size=args.batch_size,
                             num_workers=4,
                             pin_memory=True)

    # ACTUAL INFERENCE
    run_name = os.path.split(os.path.split(args.pt_ckpt_path)[0])[1]
    os.makedirs(f'test_results/{run_name}', exist_ok=True)

    eval_results = []
    for batch in tqdm(test_loader):
        if args.use_ref:
            images = (batch['src_img'], batch['ref_img'])
        else:
            images = (batch['src_img'],)
        ids = batch['id']  # [N, 1]
        gen_images, src_mask = infer_batch(generator, mask_detector, images, device)
        # evaluation first#################
        metrics = evaluate(batch['raw_gt_img'].to(device), gen_images, ssim_func,
                           ms_ssim_func)
        eval_results.append(list(metrics))
        gen_images = gen_images.cpu()
        ####################################
        gen_images = torch.split(gen_images, 1, dim=0)  # N * [1, 3, H, W]
        gen_images = [tensor2im(img.squeeze(0)) for img in gen_images]
        if src_mask is not None:
            src_mask = torch.split(src_mask, 1, dim=0)  # N * [1, H, W]
            src_mask = [tensor2im(mask.repeat((3, 1, 1))) for mask in src_mask]
        ids = ids.squeeze(1).tolist()

        for i, img in enumerate(gen_images):
            img.save(f'test_results/{run_name}/gen_{ids[i]}.jpg')
            if src_mask:
                src_mask[i].save(f'test_results/{run_name}/mask_{ids[i]}.jpg')

    # eval
    eval_results = np.array(eval_results)
    eval_results = eval_results.mean(0)
    df = pd.DataFrame({
        'ssim': [eval_results[0]],
        'ms_ssim': [eval_results[1]],
        # 'fid': eval_results[2]
    })

    print(df)

    df.to_csv(f'test_results/{run_name}/metrics.csv', index=False)


if __name__ == '__main__':
    main()
