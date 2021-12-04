from torch import nn
import torch.nn.functional as F

from modules.psp.criteria import id_loss, w_norm
from modules.psp.criteria.lpips.lpips import LPIPS
from modules.loss import VGGLoss


class pSpLoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        # Initialize loss
        self.id_lambda = args.id_lambda
        self.lpips_lambda = args.lpips_lambda
        self.l2_lambda = args.l2_lambda
        self.style_lambda = args.style_lambda

        # reference loss
        self.lpips_lambda_ref = args.lpips_lambda_ref
        self.l2_lambda_ref = args.l2_lambda_ref
        # self.cx_lambda = args.cx_lambda

        # W constraint loss
        self.w_norm_lambda = args.w_norm_lambda

        self.mse_loss = nn.MSELoss().eval()
        if self.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').eval()
        if self.id_lambda > 0:
            self.id_loss = id_loss.IDLoss().eval()
        if self.w_norm_lambda > 0:
            self.w_norm_loss = w_norm.WNormLoss(
                start_from_latent_avg=args.start_from_latent_avg)
        if self.style_lambda > 0:
            self.vgg_loss = VGGLoss()

    def style_loss(self, y_hat, src_img, src_mask):
        return self.vgg_loss(y_hat * src_mask, src_img, lossType='style')

    def contextual_loss(self, y_hat, ref_img, src_mask):
        src_mask = src_mask.unsqueeze(1)  # No inverse
        return self.vgg_loss(y_hat * src_mask, ref_img * src_mask, lossType='contextual')

    def __call__(self, x, y, y_hat, latent, latent_avg=None, ref=None, mask=None):
        loss_dict = {}
        loss = 0.0
        id_logs = None

        if mask is not None:
            mask = mask.unsqueeze(1)

        # loss wrt original image
        if self.id_lambda > 0:
            loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss = loss_id * self.id_lambda
        if self.l2_lambda > 0:
            if mask is not None:
                inv_mask = (1 - mask)
                loss_l2 = F.mse_loss(y_hat * inv_mask, y * inv_mask)
            else:
                loss_l2 = F.mse_loss(y_hat, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.l2_lambda
        if self.lpips_lambda > 0:
            if mask is not None:
                inv_mask = (1 - mask)
                loss_lpips = self.lpips_loss(y_hat * inv_mask, y * inv_mask)
            else:
                loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.lpips_lambda
        if self.style_lambda > 0 and mask is not None:
            style_loss = self.style_loss(y_hat, x, (1 - mask)) * self.style_lambda
            loss_dict['loss_style'] = float(style_loss)

        # loss wrt reference image
        if self.lpips_lambda_ref > 0 and ref is not None:
            loss_lpips_ref = self.lpips_loss(y_hat * mask, ref * mask)
            loss_dict['loss_lpips_ref'] = float(loss_lpips_ref)
            loss += loss_lpips_ref * self.lpips_lambda_ref
        if self.l2_lambda_ref > 0 and ref is not None:
            loss_l2_ref = F.mse_loss(y_hat * mask, ref * mask)
            loss_dict['loss_l2_ref'] = float(loss_l2_ref)
            loss += loss_l2_ref * self.l2_lambda_ref
        # if self.cx_lambda > 0:
        #     cx_loss = self.contextual_loss(y_hat, ref, mask) * self.cx_lambda
        #     loss_dict['loss_context'] = float(cx_loss)

        # W constraint loss
        if self.w_norm_lambda > 0 and latent_avg is not None:
            loss_w_norm = self.w_norm_loss(latent, latent_avg.to(latent.device))
            loss_dict['loss_w_norm'] = float(loss_w_norm)
            loss += loss_w_norm * self.w_norm_lambda

        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs
