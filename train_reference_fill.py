import argparse
import logging
import os
from pathlib import Path

import torch
import wandb
from torch import optim
from tqdm import tqdm

from dataloader import get_reference_dataloader
from modules.loss import GANOptimizer
from modules.mask_detector import MaskDetector
from modules.model import ReferenceFill, scale_img
from modules.pluralistic_model import base_function, network
from modules.evaluations.fid import calculate_fid
from modules.evaluations.ssim import SSIM


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

    # path args
    parser.add_argument('--run_name', type=str, default='', help='exp name')
    parser.add_argument('--checkpoint_path', type=str, default='saved_model')
    parser.add_argument('--mask_detector_path', type=str, default='')
    parser.add_argument('--data_root', type=str, default='/data/mohaa/project1/CelebA')
    parser.add_argument('--src_img_path', type=str, default='img_align_celeba_masked1')
    parser.add_argument('--ref_img_path', type=str, default='img_align_celeba')
    parser.add_argument('--mask_path', type=str, default='binary_map')
    parser.add_argument('--identity_file_path', type=str, default='identity_CelebA.txt')

    # encoder args
    parser.add_argument('--encoder_ngf', type=int, default=32, help='base filters')
    parser.add_argument('--encoder_img_f', type=int, default=128, help='final filters')
    parser.add_argument('--encoder_layers', type=int, default=5)
    parser.add_argument('--encoder_norm', type=str, default='none')
    parser.add_argument('--encoder_activation', type=str, default='LeakyReLU')
    parser.add_argument('--encoder_init_type', type=str, default='orthogonal')

    # decoder args
    parser.add_argument('--decoder_ngf', type=int, default=32, help='base filters')
    parser.add_argument('--decoder_layers', type=int, default=5)
    parser.add_argument('--decoder_norm', type=str, default='instance')
    parser.add_argument('--decoder_activation', type=str, default='LeakyReLU')
    parser.add_argument('--decoder_init_type', type=str, default='orthogonal')

    # discriminator args
    parser.add_argument('--disc_ndf', type=int, default=32, help='base filters')
    parser.add_argument('--disc_layers', type=int, default=5)
    parser.add_argument('--disc_model_type', type=str, default='ResDis')
    parser.add_argument('--disc_init_type', type=str, default='orthogonal')

    parser.add_argument('--use_att', type=int, default=1, help='whether to use attention')
    args = parser.parse_args()

    # process data path args here
    args.src_img_path = os.path.join(args.data_root, args.src_img_path)
    args.ref_img_path = os.path.join(args.data_root, args.ref_img_path)
    args.mask_path = os.path.join(args.data_root, args.mask_path)
    args.identity_file_path = os.path.join(args.data_root, args.identity_file_path)

    return args


def process_params(args):
    encoder_params = {
        k.replace('encoder_', ''): v
        for k, v in args._get_kwargs()
        if k.startswith('encoder')
    }
    decoder_params = {
        k.replace('decoder_', ''): v
        for k, v in args._get_kwargs()
        if k.startswith('decoder')
    }
    decoder_params['img_f'] = encoder_params['img_f'] * 2
    disc_params = {
        k.replace('disc_', ''): v for k, v in args._get_kwargs() if k.startswith('disc')
    }
    disc_params['img_f'] = decoder_params['img_f']
    return encoder_params, decoder_params, disc_params


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load saved mask detector
    mask_detector = MaskDetector(n_channels=3, bilinear=True)
    if args.mask_detector_path:
        mask_detector.load_state_dict(torch.load(args.mask_detector_path))
    base_function._freeze(mask_detector)  # freeze

    # process encoder, decoder, discriminator args
    encoder_params, decoder_params, disc_params = process_params(args)

    # define models
    generator = ReferenceFill(mask_detector,
                              encoder_params,
                              decoder_params,
                              use_att=args.use_att).to(device)
    discriminator = network.define_d(**disc_params).to(device)

    train_loader, val_loader = get_reference_dataloader(args.src_img_path,
                                                        args.ref_img_path,
                                                        args.mask_path,
                                                        args.identity_file_path,
                                                        args.batch_size,
                                                        val_amount=0.1,
                                                        num_workers=os.cpu_count())

    train_net(generator,
              discriminator,
              device,
              train_loader,
              val_loader,
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
             discriminator,
             val_loader,
             calc_loss,
             device,
             batch_size,
             options={'fid', 'ssim'}):
    generator.eval()
    discriminator.eval()
    num_val_batches = len(val_loader)

    metrics = {'D validation loss': 0, 'G validation loss': 0}
    if 'ssim' in options:
        ssim_loss = SSIM()  #SSIM module

    # iterate over the validation set
    for batch in tqdm(val_loader,
                      total=num_val_batches,
                      desc='Validation round',
                      unit='batch',
                      leave=False):
        src_images = batch['src_img']
        true_masks = batch['mask']
        ref_images = batch['ref_img']
        gt_images = batch['gt_img']

        src_images = src_images.to(device)  #[N, 3, H, W]
        ref_images = ref_images.to(device)  #[N, 3, H, W]
        gt_images = gt_images.to(device)  #[N, 3, H, W]
        true_masks = (true_masks > 0).float().to(device)  #[N, H, W]

        gen_images = generator(src_images, ref_images, src_mask=true_masks)  #[N, 3, H, W]

        # calculate metrics
        if 'fid' in options:
            fid_distance = calculate_fid(scale_img(gt_images, (299, 299)),
                                         scale_img(gen_images, (299, 299)), False,
                                         batch_size)
            if 'fid' in metrics:
                metrics['fid'] += fid_distance
            else:
                metrics['fid'] = fid_distance

        if 'ssim' in options:
            ssim = ssim_loss(gt_images, gen_images)
            if 'ssim' in metrics:
                metrics['ssim'] += ssim
            else:
                metrics['ssim'] = ssim

        # now loss
        loss_D, loss_G = calc_loss(discriminator, src_images, gt_images, ref_images,
                                   gen_images, true_masks)
        metrics['D validation loss'] += loss_D.item()
        metrics['G validation loss'] += loss_G.item()

    generator.train()
    discriminator.train()

    metrics = {k: v / num_val_batches for k, v in metrics.items()}
    return metrics


def train_net(generator,
              discriminator,
              device,
              train_loader,
              val_loader,
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

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G,
                                                       'max',
                                                       patience=2,
                                                       factor=0.8)

    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D,
                                                       'max',
                                                       patience=2,
                                                       factor=0.8)

    gan_optimizer = GANOptimizer(optimizer_D, optimizer_G, debug=debug).to(device)
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        epoch_loss_D = 0
        epoch_loss_G = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                src_images = batch['src_img']
                true_masks = batch['mask']
                ref_images = batch['ref_img']
                gt_images = batch['gt_img']

                src_images = src_images.to(device)  #[N, 3, H, W]
                ref_images = ref_images.to(device)  #[N, 3, H, W]
                gt_images = gt_images.to(device)  #[N, 3, H, W]
                true_masks = (true_masks > 0).float().to(device)  #[N, H, W]

                gen_images = generator(src_images, ref_images, src_mask=true_masks)

                loss_D, loss_G = gan_optimizer(discriminator, src_images, gt_images,
                                               ref_images, gen_images, true_masks)

                pbar.update(src_images.shape[0])
                global_step += 1
                epoch_loss_D += loss_D.item()
                epoch_loss_G += loss_G.item()
                experiment.log({
                    'G train loss': loss_G.item(),
                    'D train loss': loss_D.item(),
                    'step': global_step,
                    'epoch': epoch,
                })
                pbar.set_postfix(**{
                    'G loss (batch)': loss_G.item(),
                    'D loss (batch)': loss_D.item(),
                })

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
                    for tag, value in discriminator.named_parameters():
                        if value.grad is None:  # auto_attn grad is None
                            continue
                        tag = tag.replace('/', '.')
                        histograms['D_weights/' + tag] = wandb.Histogram(value.data.cpu())
                        histograms['D_gradients/' + tag] = wandb.Histogram(
                            value.grad.data.cpu())

                    exp_log_params = {
                        '[G] learning rate': optimizer_G.param_groups[0]['lr'],
                        '[D] learning rate': optimizer_D.param_groups[0]['lr'],
                        'src_images': wandb.Image(src_images[0].cpu()),
                        'ref_images': wandb.Image(ref_images[0].cpu()),
                        'gen_images': wandb.Image(gen_images[0].cpu()),
                        'gt_images': wandb.Image(gt_images[0].cpu()),
                        'step': global_step,
                        'epoch': epoch,
                        **histograms
                    }
                    if len(eval_options) > 0:
                        metrics = evaluate(generator, discriminator, val_loader,
                                           gan_optimizer.calc_loss, device, batch_size,
                                           eval_options)
                        scheduler_D.step(metrics['D validation loss'])
                        scheduler_G.step(metrics['G validation loss'])
                        for k, v in metrics.items():
                            logging.info(f'{k}: {v}')
                            exp_log_params[k] = v
                    experiment.log(exp_log_params)

        if save_checkpoint:
            torch.save(generator.state_dict(),
                       dir_checkpoint / Path(f'G_checkpoint_epoch{epoch + 1}.pth'))
            torch.save(discriminator.state_dict(),
                       dir_checkpoint / Path(f'D_checkpoint_epoch{epoch + 1}.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved!')


if __name__ == '__main__':
    main()
