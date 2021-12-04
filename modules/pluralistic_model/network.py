from modules.pluralistic_model.base_function import *
from modules.pluralistic_model.external_function import SpectralNorm


##############################################################################################################
# Network function
##############################################################################################################
def define_e(input_nc=3,
             ngf=64,
             img_f=512,
             layers=5,
             norm='none',
             activation='ReLU',
             use_spect=True,
             use_coord=False,
             init_type='orthogonal',
             gpu_ids=[]):

    net = ResEncoder(input_nc, ngf, img_f, layers, norm, activation, use_spect, use_coord)

    return init_net(net, init_type, activation, gpu_ids)


def define_g(output_nc=3,
             ngf=64,
             img_f=512,
             layers=5,
             norm='instance',
             activation='ReLU',
             use_spect=True,
             use_coord=False,
             use_attn=True,
             init_type='orthogonal',
             gpu_ids=[]):

    net = ResGenerator(output_nc, ngf, img_f, layers, norm, activation, use_spect,
                       use_coord, use_attn)

    return init_net(net, init_type, activation, gpu_ids)


def define_d(input_nc=3,
             ndf=64,
             img_f=512,
             layers=6,
             norm='none',
             activation='LeakyReLU',
             use_spect=True,
             use_coord=False,
             use_attn=True,
             model_type='ResDis',
             init_type='orthogonal',
             gpu_ids=[]):

    if model_type == 'ResDis':
        net = ResDiscriminator(input_nc, ndf, img_f, layers, norm, activation, use_spect,
                               use_coord, use_attn)
    elif model_type == 'PatchDis':
        net = PatchDiscriminator(input_nc, ndf, img_f, layers, norm, activation,
                                 use_spect, use_coord, use_attn)

    return init_net(net, init_type, activation, gpu_ids)


#############################################################################################################
# Network structure
#############################################################################################################
class ResEncoder(nn.Module):
    """
    ResNet Encoder Network
    :param input_nc: number of channels in input
    :param ngf: base filter channel
    :param img_f: the largest feature channels
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """

    def __init__(self,
                 input_nc=3,
                 ngf=64,
                 img_f=1024,
                 layers=6,
                 norm='none',
                 activation='ReLU',
                 use_spect=True,
                 use_coord=False):
        super(ResEncoder, self).__init__()

        self.layers = layers

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ngf, norm_layer, nonlinearity,
                                               use_spect, use_coord)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2**(i + 1), img_f // ngf)
            if i % 2 == 0:
                block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer,
                                 nonlinearity, 'none', use_spect, use_coord)
            else:
                block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer,
                                 nonlinearity, 'down', use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

    def forward(self, img):
        """
        :param img: image with mask regions I_m
        :return feature: the conditional feature f_m, and the previous f_pre for auto context attention
        """
        # encoder part
        out = self.block0(img)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)

        return out


class ResGenerator(nn.Module):
    """
    ResNet Generator Network
    :param output_nc: number of channels in output
    :param ngf: base filter channel
    :param img_f: the largest feature channels
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """

    def __init__(self,
                 output_nc=3,
                 ngf=64,
                 img_f=1024,
                 layers=6,
                 norm='batch',
                 activation='ReLU',
                 use_spect=True,
                 use_coord=False,
                 use_attn=True):
        super(ResGenerator, self).__init__()

        self.layers = layers
        self.use_attn = use_attn

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        # decoder part
        mult = min(2**(layers - 1), img_f // ngf)
        for i in range(layers):
            mult_prev = mult
            mult = min(2**(layers - i - 1), img_f // ngf)
            upconv = ResBlockDecoder(ngf * mult_prev, ngf * mult, ngf * mult, norm_layer,
                                     nonlinearity, use_spect, use_coord)
            # # actually not an upconv!
            # upconv = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev,
            #                     norm_layer, nonlinearity, 'none', use_spect, use_coord)
            setattr(self, 'decoder' + str(i), upconv)
            # output part
            if i > layers - 2:
                outconv = Output(ngf * mult, output_nc, 3, None, nonlinearity, use_spect,
                                 use_coord)
                setattr(self, 'out' + str(i), outconv)

    def forward(self, encoded):
        """
        ResNet Generator Network
        :param f_m: feature of valid regions for conditional VAG-GAN
        :return results: different scale generation outputs
        """

        out = encoded
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)
            if i > self.layers - 2:
                model = getattr(self, 'out' + str(i))
                output = model(out)
                out = torch.cat([out, output], dim=1)
        return output


class ResDiscriminator(nn.Module):
    """
    ResNet Discriminator Network
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param layers: down and up sample layers
    :param img_f: the largest feature channels
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """

    def __init__(self,
                 input_nc=3,
                 ndf=64,
                 img_f=1024,
                 layers=6,
                 norm='none',
                 activation='LeakyReLU',
                 use_spect=True,
                 use_coord=False,
                 use_attn=True):
        super(ResDiscriminator, self).__init__()

        self.layers = layers
        self.use_attn = use_attn

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.nonlinearity = nonlinearity

        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ndf, norm_layer, nonlinearity,
                                               use_spect, use_coord)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2**(i + 1), img_f // ndf)
            # self-attention
            if i == 2 and use_attn:
                attn = Auto_Attn(ndf * mult_prev, norm_layer)
                setattr(self, 'attn' + str(i), attn)
            block = ResBlock(ndf * mult_prev, ndf * mult, ndf * mult_prev, norm_layer,
                             nonlinearity, 'down', use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)

        self.block1 = ResBlock(ndf * mult, ndf * mult, ndf * mult, norm_layer,
                               nonlinearity, 'none', use_spect, use_coord)
        self.conv = SpectralNorm(nn.Conv2d(ndf * mult, 1, 3))

    def forward(self, x):
        out = self.block0(x)
        for i in range(self.layers - 1):
            if i == 2 and self.use_attn:
                attn = getattr(self, 'attn' + str(i))
                out, attention = attn(out)
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out = self.block1(out)
        out = self.conv(self.nonlinearity(out))  # [N, 1, H, W]
        out = out.reshape(out.shape[0], 1, -1)
        return out.mean(-1)  # [N, 1]


class PatchDiscriminator(nn.Module):
    """
    Patch Discriminator Network for Local 70*70 fake/real
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param img_f: the largest channel for the model
    :param layers: down sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectral normalization or not
    :param use_coord: use CoordConv or nor
    :param use_attn: use short+long attention or not
    """

    def __init__(self,
                 input_nc=3,
                 ndf=64,
                 img_f=512,
                 layers=3,
                 norm='batch',
                 activation='LeakyReLU',
                 use_spect=True,
                 use_coord=False,
                 use_attn=False):
        super(PatchDiscriminator, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        kwargs = {'kernel_size': 4, 'stride': 2, 'padding': 1, 'bias': False}
        sequence = [
            coord_conv(input_nc, ndf, use_spect, use_coord, **kwargs),
            nonlinearity,
        ]

        mult = 1
        for i in range(1, layers):
            mult_prev = mult
            mult = min(2**i, img_f // ndf)
            sequence += [
                coord_conv(ndf * mult_prev, ndf * mult, use_spect, use_coord, **kwargs),
                nonlinearity,
            ]

        mult_prev = mult
        mult = min(2**i, img_f // ndf)
        kwargs = {'kernel_size': 4, 'stride': 1, 'padding': 1, 'bias': False}
        sequence += [
            coord_conv(ndf * mult_prev, ndf * mult, use_spect, use_coord, **kwargs),
            nonlinearity,
            coord_conv(ndf * mult, 1, use_spect, use_coord, **kwargs),
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        return out
