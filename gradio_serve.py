import argparse

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from modules.mask_detector import MaskDetector
from modules.psp.psp import pSp
from modules.model import scale_img


class ModelInterface:

    def __init__(self, args, device):
        self.mask_detector = MaskDetector(n_channels=3, bilinear=True).to(device)
        self.mask_detector.load_state_dict(torch.load(args.mask_detector_path))
        # define models
        self.generator = pSp(args).to(device)
        if self.generator.latent_avg is None:
            self.generator.latent_avg = self.generator.decoder.mean_latent(
                int(1e5))[0].detach()

        self.device = device
        self.mask_detector.eval()
        self.generator.eval()

        self.transform = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def preprocess_img(self, img):
        org_size = img.size
        org_size = (org_size[1], org_size[0])
        img = img.resize((256, 256), resample=Image.BICUBIC)
        img = np.asarray(img)

        if img.ndim == 2:
            img = img[np.newaxis, ...]
        img = img.transpose((2, 0, 1))
        img = img / 255
        img = torch.as_tensor(img.copy()).float().contiguous()
        img = self.transform(img).unsqueeze(0)
        return img, org_size

    def tensor2im(self, img):
        img = img.permute(1, 2, 0).numpy()
        img[img < 0] = 0
        img[img > 1] = 1
        img = img * 255
        img = img.astype('uint8')
        return img

    def infer(self, src_img, ref_img):
        src_img, src_size = self.preprocess_img(src_img)  # [1, 3, H, W]
        src_img = src_img.to(self.device)
        ref_img, _ = self.preprocess_img(ref_img)  # [1, 3, H, W]
        ref_img = ref_img.to(self.device)
        gen_img, mask = self.infer_image(src_img, ref_img)
        gen_img = (gen_img + 1) / 2
        mask = mask.repeat(3, 1, 1)  # [3, H, W]
        gen_img = scale_img(gen_img, src_size)
        mask = scale_img(mask.unsqueeze(0), src_size).squeeze(0)
        gen_img, mask = self.tensor2im(gen_img[0]), self.tensor2im(mask)
        return gen_img, mask

    @torch.no_grad()
    def infer_image(self, src_img, ref_img):
        src_mask = self.mask_detector((src_img + 1) / 2, mode='train')  # [1, 2, H, W]
        src_mask = src_mask.argmax(1).float()  # [1, H, W]

        gen_images = self.generator(src_img,
                                    ref=ref_img,
                                    src_mask=src_mask,
                                    resize=True,
                                    randomize_noise=False)  # [1, 3, H, W]

        return gen_images.detach().cpu(), src_mask.detach().cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pt_ckpt_path',
        default='saved_model/RefpSp_train_decoder/G_checkpoint_epoch5.pth',
        type=str,
        help='Path to pretrained pSp model checkpoint')
    parser.add_argument('--mask_detector_path',
                        default='saved_model/new_mask_detector.pth',
                        type=str,
                        help='Path to pretrained pSp model checkpoint')
    parser.add_argument('--use_attention', default=0, type=int, help='use attention')

    # pSp args: DO NOT MODIFY
    parser.add_argument('--use_ref', default=1, type=int, help='use reference image')
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
    # pretrained weight paths
    parser.add_argument('--stylegan_weights',
                        default=None,
                        type=str,
                        help='Path to StyleGAN model weights')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelInterface(args, device)

    iface = gr.Interface(fn=model.infer,
                         inputs=[
                             gr.inputs.Image(shape=None,
                                             image_mode="RGB",
                                             source="upload",
                                             tool="editor",
                                             type="pil",
                                             label="Image with mask",
                                             optional=False),
                             gr.inputs.Image(shape=None,
                                             image_mode="RGB",
                                             source="upload",
                                             tool="editor",
                                             type="pil",
                                             label="Image of the same person",
                                             optional=False)
                         ],
                         outputs=[
                             gr.outputs.Image(type="auto", label="Unmasked Image"),
                             gr.outputs.Image(type="auto", label="Mask Region"),
                         ],
                         title='Remove Facial Mask Demo',
                         theme='huggingface')
    iface.launch(share=True)


if __name__ == '__main__':
    main()
