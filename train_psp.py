import argparse
import logging
import os
from pathlib import Path

import torch
import wandb
from torch import optim
from tqdm import tqdm

from dataloader import get_reference_dataloader
from modules.mask_detector import MaskDetector
from modules.model import scale_img
from modules.pluralistic_model import base_function
from modules.evaluations.fid import calculate_fid
from modules.evaluations.ssim import SSIM

# psp
from modules.psp.psp import pSp
from modules.psp.criteria import pSpLoss
from modules.psp.ranger import Ranger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--eval_options', nargs="+", default={'ssim'})
    parser.add_argument('--debug',
                        type=int,
                        default=0,
                        help='debug with turning off not implemented parts')
    parser.add_argument('--img_scale', type=float, default=1.)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--use_ref', action='store_true', help='use reference image')

    # path args
    parser.add_argument('--run_name', type=str, default='', help='exp name')
    parser.add_argument('--checkpoint_path', type=str, default='saved_model')
    parser.add_argument('--mask_detector_path', type=str, default='')
    parser.add_argument('--data_root', type=str, default='/data/mohaa/project1/CelebA')
    parser.add_argument('--src_img_path', type=str, default='img_align_celeba_masked1')
    parser.add_argument('--ref_img_path', type=str, default='img_align_celeba')
    parser.add_argument('--mask_path', type=str, default='binary_map')
    parser.add_argument('--identity_file_path', type=str, default='identity_CelebA.txt')

    # pSp args
    parser.add_argument('--encoder_type', type=str, default='GradualStyleEncoder')
    parser.add_argument('--output_size',
                        default=1024,
                        type=int,
                        help='Output size of generator')
    parser.add_argument('--train_decoder',
                        default=False,
                        type=bool,
                        help='Whether to train the decoder model')
    parser.add_argument(
        '--start_from_latent_avg',
        action='store_true',
        help='Whether to add average latent vector to generate codes from encoder.')
    parser.add_argument('--learn_in_w',
                        action='store_true',
                        help='Whether to learn in w space instead of w+')
    parser.add_argument('--randomize_noise',
                        action='store_true',
                        help='whether to randomize noise in stylegan')

    # loss weights
    parser.add_argument('--lpips_lambda',
                        default=0.8,
                        type=float,
                        help='LPIPS loss multiplier factor')
    parser.add_argument('--id_lambda',
                        default=0,
                        type=float,
                        help='ID loss multiplier factor')
    parser.add_argument('--l2_lambda',
                        default=1.0,
                        type=float,
                        help='L2 loss multiplier factor')
    parser.add_argument('--w_norm_lambda',
                        default=0,
                        type=float,
                        help='W-norm loss multiplier factor')
    parser.add_argument('--lpips_lambda_crop',
                        default=0,
                        type=float,
                        help='LPIPS loss multiplier factor for inner image region')
    parser.add_argument('--l2_lambda_crop',
                        default=0,
                        type=float,
                        help='L2 loss multiplier factor for inner image region')
    parser.add_argument('--moco_lambda', default=0, type=float, help='moco_lambda')
    parser.add_argument('--style_lambda', default=250, type=float)

    # pretrained weight paths
    parser.add_argument('--stylegan_weights',
                        default=None,
                        type=str,
                        help='Path to StyleGAN model weights')
    parser.add_argument('--pt_ckpt_path',
                        default=None,
                        type=str,
                        help='Path to pretrained pSp model checkpoint')
    args = parser.parse_args()

    # process data path args here
    args.src_img_path = os.path.join(args.data_root, args.src_img_path)
    args.ref_img_path = os.path.join(args.data_root, args.ref_img_path)
    args.mask_path = os.path.join(args.data_root, args.mask_path)
    args.identity_file_path = os.path.join(args.data_root, args.identity_file_path)

    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load saved mask detector
    mask_detector = MaskDetector(n_channels=3, bilinear=True)
    if args.mask_detector_path:
        mask_detector.load_state_dict(torch.load(args.mask_detector_path))
    base_function._freeze(mask_detector)  # freeze

    # define models
    generator = pSp(args).to(device)
    if generator.latent_avg is None:
        generator.latent_avg = generator.decoder.mean_latent(int(1e5))[0].detach()

    train_loader, val_loader = get_reference_dataloader(args.src_img_path,
                                                        args.ref_img_path,
                                                        args.mask_path,
                                                        args.identity_file_path,
                                                        args.batch_size,
                                                        val_amount=0.1,
                                                        num_workers=os.cpu_count(),
                                                        img_scale=args.img_scale,
                                                        use_ssim=True)

    train_net(generator,
              device,
              train_loader,
              val_loader,
              args,
              epochs=args.epochs,
              batch_size=args.batch_size,
              learning_rate=args.learning_rate,
              save_checkpoint=True,
              dir_checkpoint=args.checkpoint_path,
              run_name=args.run_name,
              eval_options=set(args.eval_options),
              debug=bool(args.debug))


@torch.no_grad()
def evaluate(generator,
             val_loader,
             calc_loss,
             device,
             batch_size,
             latent_avg=None,
             use_ref=True,
             options={'fid', 'ssim'}):
    generator.eval()
    num_val_batches = len(val_loader)

    metrics = {'val loss': 0}
    if 'ssim' in options:
        ssim_loss = SSIM()  #SSIM module

    # iterate over the validation set
    for batch in tqdm(val_loader,
                      total=num_val_batches,
                      desc='Validation round',
                      unit='batch',
                      leave=False):
        src_images = batch['src_img']
        gt_images = batch['gt_img']
        raw_gt_img = batch['raw_gt_img']

        src_images = src_images.to(device)  #[N, 3, H, W]
        gt_images = gt_images.to(device)  #[N, 3, H, W]
        raw_gt_img = raw_gt_img.to(device)
        if use_ref:
            ref_images = batch['ref_img'].to(device)  # [N, 3, H, W]
            true_masks = (batch['mask'] > 0).float().to(device)  # [N, H, W]
            ref_images = ref_images
            # src_images = (1 - true_masks).unsqueeze(1) * src_images  # corrupt images
        else:
            ref_images = true_masks = None
        gen_images, latent = generator(src_images,
                                       ref=ref_images,
                                       src_mask=true_masks,
                                       return_latents=True)  #[N, 3, H, W]

        # now loss
        loss, _, _ = calc_loss(src_images,
                               gt_images,
                               gen_images,
                               latent,
                               latent_avg=latent_avg,
                               ref=ref_images,
                               mask=true_masks)
        metrics['val loss'] += loss.item()

        gen_images = (gen_images + 1) / 2
        # calculate metrics
        if 'fid' in options:
            fid_distance = calculate_fid(scale_img(raw_gt_img, (299, 299)),
                                         scale_img(gen_images, (299, 299)), False,
                                         batch_size)
            if 'fid' in metrics:
                metrics['fid'] += fid_distance
            else:
                metrics['fid'] = fid_distance

        if 'ssim' in options:
            ssim = ssim_loss(raw_gt_img, gen_images)
            if 'ssim' in metrics:
                metrics['ssim'] += ssim
            else:
                metrics['ssim'] = ssim

    generator.train()

    metrics = {k: v / num_val_batches for k, v in metrics.items()}
    return metrics


def train_net(generator,
              device,
              train_loader,
              val_loader,
              args,
              epochs=5,
              batch_size=1,
              learning_rate=0.001,
              save_checkpoint=True,
              dir_checkpoint=None,
              run_name='',
              eval_options={'ssim', 'fid'},
              debug=False):

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)

    experiment = wandb.init(project='Reference Inpainting',
                            resume='allow',
                            name=run_name,
                            anonymous='must')
    experiment.config.update(
        dict(epochs=epochs,
             batch_size=batch_size,
             learning_rate=learning_rate,
             save_checkpoint=save_checkpoint))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device}
    ''')

    if isinstance(dir_checkpoint, str):
        dir_checkpoint = Path(dir_checkpoint)
    dir_checkpoint = dir_checkpoint / Path(run_name)
    dir_checkpoint.mkdir(parents=True, exist_ok=True)

    if args.optimizer == 'adam':
        optimizer = optim.Adam(generator.encoder.parameters(), lr=learning_rate)
    elif args.optimizer == 'ranger':
        optimizer = Ranger(generator.encoder.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     'max',
                                                     patience=2,
                                                     factor=0.8)

    global_step = 0

    psp_loss = pSpLoss(args).to(device)
    # 5. Begin training
    for epoch in range(epochs):
        generator.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                src_images = batch['src_img'].to(device)  # [N, 3, H, W]
                gt_images = batch['gt_img'].to(device)  # [N, 3, H, W]
                if args.use_ref:
                    ref_images = batch['ref_img'].to(device)  # [N, 3, H, W]
                    true_masks = (batch['mask'] > 0).float().to(device)  # [N, H, W]
                    # src_images = (1 - true_masks).unsqueeze(1) * src_images  # corrupt images
                else:
                    ref_images = true_masks = None
                gen_images, latent = generator(src_images,
                                               ref=ref_images,
                                               src_mask=true_masks,
                                               return_latents=True,
                                               randomize_noise=args.randomize_noise)
                loss, loss_dict, id_logs = psp_loss(src_images,
                                                    gt_images,
                                                    gen_images,
                                                    latent,
                                                    latent_avg=generator.latent_avg,
                                                    ref=ref_images,
                                                    mask=true_masks)
                if not torch.isfinite(loss):
                    # TODO: The batch['gt_img'].to(device) makes problem:
                    # the values get changed to very large number
                    pass
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.update(src_images.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                    experiment.log({
                        **loss_dict,
                        'step': global_step,
                        'epoch': epoch,
                    })
                    pbar.set_postfix(**loss_dict)

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step == 0:
                    continue
                if global_step % division_step == 0:
                    histograms = {}
                    for tag, value in generator.named_parameters():
                        if not value.requires_grad:
                            continue
                        tag = tag.replace('/', '.')
                        histograms['G_weights/' + tag] = wandb.Histogram(value.data.cpu())
                        histograms['G_gradients/' + tag] = wandb.Histogram(
                            value.grad.data.cpu())

                    exp_log_params = {
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'src_images': wandb.Image(src_images[0].cpu()),
                        'gen_images': wandb.Image(gen_images[0].cpu()),
                        'gt_images': wandb.Image(gt_images[0].cpu()),
                        'step': global_step,
                        'epoch': epoch,
                        **histograms
                    }
                    if ref_images is not None:
                        exp_log_params['ref_images'] = wandb.Image(ref_images[0].cpu())
                    if len(eval_options) > 0:
                        metrics = evaluate(generator,
                                           val_loader,
                                           psp_loss,
                                           device,
                                           batch_size,
                                           use_ref=args.use_ref,
                                           latent_avg=generator.latent_avg,
                                           options=eval_options)
                        scheduler.step(metrics['val loss'])
                        for k, v in metrics.items():
                            logging.info(f'{k}: {v}')
                            exp_log_params[k] = v
                    experiment.log(exp_log_params)

        if save_checkpoint:
            torch.save(generator.state_dict(),
                       dir_checkpoint / Path(f'G_checkpoint_epoch{epoch + 1}.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved!')


if __name__ == '__main__':
    main()
