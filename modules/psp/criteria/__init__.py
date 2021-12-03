from torch import nn
import torch.nn.functional as F

from modules.psp.criteria import id_loss, w_norm, moco_loss
from modules.psp.criteria.lpips.lpips import LPIPS
from modules.loss import VGGLoss


class pSpLoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        # Initialize loss
        self.id_lambda = args.id_lambda
        self.moco_lambda = args.moco_lambda
        self.lpips_lambda = args.lpips_lambda
        self.w_norm_lambda = args.w_norm_lambda
        self.l2_lambda = args.l2_lambda
        self.lpips_lambda_crop = args.lpips_lambda_crop
        self.l2_lambda_crop = args.l2_lambda_crop
        self.style_lambda = args.style_lambda
        # self.cx_lambda = args.cx_lambda
        if self.id_lambda > 0 and self.moco_lambda > 0:
            raise ValueError(
                'Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!'
            )

        self.mse_loss = nn.MSELoss().eval()
        if self.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').eval()
        if self.id_lambda > 0:
            self.id_loss = id_loss.IDLoss().eval()
        if self.w_norm_lambda > 0:
            self.w_norm_loss = w_norm.WNormLoss(
                start_from_latent_avg=args.start_from_latent_avg)
        if self.moco_lambda > 0:
            self.moco_loss = moco_loss.MocoLoss().eval()
        if self.style_lambda > 0:
            self.vgg_loss = VGGLoss()

    def style_loss(self, y_hat, src_img, src_mask):
        src_mask = (1 - src_mask).unsqueeze(1)  # Yes inverse
        return self.vgg_loss(y_hat * src_mask, src_img, lossType='style')

    def contextual_loss(self, y_hat, ref_img, src_mask):
        src_mask = src_mask.unsqueeze(1)  # No inverse
        return self.vgg_loss(y_hat * src_mask, ref_img * src_mask, lossType='contextual')

    def __call__(self, x, y, y_hat, latent, latent_avg=None, ref=None, mask=None):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.id_lambda > 0:
            loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss = loss_id * self.id_lambda
        if self.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.l2_lambda
        if self.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.lpips_lambda
        if self.lpips_lambda_crop > 0:
            loss_lpips_crop = self.lpips_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223,
                                                                             32:220])
            loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
            loss += loss_lpips_crop * self.lpips_lambda_crop
        if self.l2_lambda_crop > 0:
            loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223,
                                                                     32:220])
            loss_dict['loss_l2_crop'] = float(loss_l2_crop)
            loss += loss_l2_crop * self.l2_lambda_crop
        if self.w_norm_lambda > 0 and latent_avg is not None:
            loss_w_norm = self.w_norm_loss(latent, latent_avg.to(latent.device))
            loss_dict['loss_w_norm'] = float(loss_w_norm)
            loss += loss_w_norm * self.w_norm_lambda
        if self.moco_lambda > 0:
            loss_moco, sim_improvement, id_logs = self.moco_loss(y_hat, y, x)
            loss_dict['loss_moco'] = float(loss_moco)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_moco * self.moco_lambda
        if self.style_lambda > 0:
            style_loss = self.style_loss(y_hat, x, mask) * self.style_lambda
            loss_dict['loss_style'] = float(style_loss)
        # if self.cx_lambda > 0:
        #     cx_loss = self.contextual_loss(y_hat, ref, mask) * self.cx_lambda
        #     loss_dict['loss_context'] = float(cx_loss)

        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs
