import imp
from grpc import channel_ready_future
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torchvision
from torchsummary import summary
from .rsunet import RSUNet
###############################################################################
# Helper Functions
###############################################################################

# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        # self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.weights = [1.0, 0.0, 0.0, 0.0, 0.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
        # norm_layer = nn.BatchNorm3d
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    print('netG',netG)
    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        # net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        net = UnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        # net = UnetGenerator_default(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout) # default 8
    elif netG == 'unet_256_4':
        net = UnetGenerator_4(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256_3':
        net = UnetGenerator_3(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_1133333':
        net = UnetGenerator_1133333(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_133333':
        net = UnetGenerator_133333(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_3333113':
        net = UnetGenerator_3333113(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_333313':
        net = UnetGenerator_333313(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_1113333':
        net = UnetGenerator_1113333(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_3331113':
        net = UnetGenerator_3331113(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_11111':
        net = UnetGenerator_11111(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_111111':
        net = UnetGenerator_111111(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_1111111':
        net = UnetGenerator_1111111(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'rsunet_wres':
        net = RSUNet(input_nc, output_nc, 4, residual=True)
    elif netG == 'rsunet_wores':
        net = RSUNet(input_nc, output_nc, 4, residual=False)
    elif netG == 'unet_5133':
        net = UnetGenerator_5133(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_6133':
        net = UnetGenerator_6133(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_7133':
        net = UnetGenerator_7133(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_3d':
        net = unet_3d(input_nc, output_nc)
    elif netG == 'unet_3d_k3':
        net = unet_3d_k3(input_nc, output_nc)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_E(input_nc, output_nc, ndf, netE,
             norm='batch', nl='lrelu',
             init_type='xavier', init_gain=0.02, gpu_ids=[], vaeLike=False):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl = 'lrelu'  # use leaky relu for E
    nl_layer = get_non_linearity(layer_type=nl)
    if netE == 'resnet_128':
        net = E_ResNet(input_nc, output_nc, ndf, n_blocks=4, norm_layer=norm_layer,
                       nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'resnet_256':
        net = E_ResNet(input_nc, output_nc, ndf, n_blocks=5, norm_layer=norm_layer,
                       nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'conv_128':
        net = E_NLayers(input_nc, output_nc, ndf, n_layers=4, norm_layer=norm_layer,
                        nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'conv_256':
        net = E_NLayers(input_nc, output_nc, ndf, n_layers=5, norm_layer=norm_layer,
                        nl_layer=nl_layer, vaeLike=vaeLike)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % net)

    return init_net(net, init_type, init_gain, gpu_ids)

def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer

class E_ResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_blocks=4,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_ResNet, self).__init__()
        self.vaeLike = vaeLike
        max_ndf = 4
        conv_layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf,
                                       output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer(), nn.AvgPool2d(8)]
        if vaeLike:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
            self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        else:
            return output
        return output


class E_NLayers(nn.Module):
    def __init__(self, input_nc, output_nc=1, ndf=64, n_layers=3,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_NLayers, self).__init__()
        self.vaeLike = vaeLike

        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nl_layer()]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 4)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw)]
            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_mult)]
            sequence += [nl_layer()]
        sequence += [nn.AvgPool2d(8)]
        self.conv = nn.Sequential(*sequence)
        self.fc = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_nc)])
        if vaeLike:
            self.fcVar = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_nc)])

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        return output


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)

def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes,
                           kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)

def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out

##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        return loss

class DiscLossWGANGP():
    def __init__(self):
        self.LAMBDA = 10
        
    def name(self):
        return 'DiscLossWGAN-GP'

    def initialize(self, opt, tensor):
        # DiscLossLS.initialize(self, opt, tensor)
        self.LAMBDA = 10
        
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)
        
        disc_interpolates = netD.forward(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.InstanceNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True, norm_layer=norm_layer)
        num_downs = 5
        # for i in range(num_downs - 5):
        #     unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)
        # UnetSkipConnection_nonzdownsaple_Block()
        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class UnetGenerator_4(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.InstanceNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator_4, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 4, innermost=True, norm_layer=norm_layer)
        # for i in range(num_downs - 5):
        #     unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)
        # UnetSkipConnection_nonzdownsaple_Block()
        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class UnetGenerator_3(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.InstanceNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator_3, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 4, innermost=True, norm_layer=norm_layer)
        # for i in range(num_downs - 5):
        #     unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)
        # UnetSkipConnection_nonzdownsaple_Block()
        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class UnetGenerator_1133333(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.InstanceNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator_1133333, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True, norm_layer=norm_layer)
        num_downs = 7
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_nonzdownsaple_Block(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)
        # UnetSkipConnection_nonzdownsaple_Block()
        self.model = unet_block
        # print(self.model)
        # summary(self.model, (1, 32, 256, 256), batch_size=1)
    def forward(self, input):
        return self.model(input)

class UnetGenerator_133333(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.InstanceNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator_133333, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True, norm_layer=norm_layer)
        num_downs = 6
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_nonzdownsaple_Block(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)
        # UnetSkipConnection_nonzdownsaple_Block()
        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class UnetGenerator_333313(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.InstanceNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator_333313, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True, norm_layer=norm_layer)
        num_downs = 6
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)
        # UnetSkipConnection_nonzdownsaple_Block()
        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class UnetGenerator_3333113(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.InstanceNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator_3333113, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True, norm_layer=norm_layer)
        num_downs = 7
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)
        # UnetSkipConnection_nonzdownsaple_Block()
        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class UnetGenerator_1113333(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.InstanceNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator_1113333, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True, norm_layer=norm_layer)
        num_downs = 7
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_nonzdownsaple_Block(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)
        # UnetSkipConnection_nonzdownsaple_Block()
        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class UnetGenerator_3331113(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.InstanceNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator_3331113, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True, norm_layer=norm_layer)
        num_downs = 7
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)
        # UnetSkipConnection_nonzdownsaple_Block()
        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class UnetGenerator_11111(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.InstanceNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator_11111, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf * 8, ngf * 8, innermost=True, norm_layer=norm_layer)
        # num_downs = 7
        # for i in range(num_downs - 5):
        #     unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_nonzdownsaple_Block(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)
        # UnetSkipConnection_nonzdownsaple_Block()
        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class UnetGenerator_111111(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.InstanceNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator_111111, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf * 8, ngf * 8, innermost=True, norm_layer=norm_layer)
        num_downs = 6
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_nonzdownsaple_Block(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)
        # UnetSkipConnection_nonzdownsaple_Block()
        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class UnetGenerator_1111111(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.InstanceNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator_1111111, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf * 8, ngf * 8, innermost=True, norm_layer=norm_layer)
        num_downs = 7
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_nonzdownsaple_Block(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_nonzdownsaple_Block(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)
        # UnetSkipConnection_nonzdownsaple_Block()
        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class UnetGenerator_5133(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.InstanceNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator_5133, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnection_133_Block(ngf * 8, ngf * 8, innermost=True, norm_layer=norm_layer)
        # num_downs = 7
        # for i in range(num_downs - 5):
        #     unet_block = UnetSkipConnection_133_Block(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnection_133_Block(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_133_Block(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_133_Block(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_133_Block(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)
        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class UnetGenerator_6133(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.InstanceNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator_6133, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnection_133_Block(ngf * 8, ngf * 8, innermost=True, norm_layer=norm_layer)
        num_downs = 6
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnection_133_Block(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnection_133_Block(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_133_Block(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_133_Block(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_133_Block(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)
        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class UnetGenerator_7133(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.InstanceNorm3d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator_7133, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnection_133_Block(ngf * 8, ngf * 8, innermost=True, norm_layer=norm_layer)
        num_downs = 7
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnection_133_Block(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnection_133_Block(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_133_Block(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock(ngf, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_133_Block(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnection_133_Block(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)
        self.model = unet_block

    def forward(self, input):
        return self.model(input)

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm3d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()  
        self.outermost = outermost
        self.submodule = submodule

        downconv1 = nn.Conv3d(outer_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1)
        downrelu1 = nn.ReLU(True)

        downconv2 = nn.Conv3d(inner_nc, inner_nc, kernel_size=3, padding=1)
        downrelu2 = nn.ReLU(True)

        downnorm1 = norm_layer(inner_nc, affine=True)
        downnorm2 = norm_layer(inner_nc, affine=True)
        uprelu1 = nn.ReLU(True)
        upnorm1 = norm_layer(outer_nc, affine=True)
        upconv2 = nn.Conv3d(outer_nc, outer_nc,
                                    kernel_size=3, stride=1,
                                    padding=1)
        upnorm2 = norm_layer(outer_nc, affine=True)
        uprelu2 = nn.ReLU(True)
        
        if outermost:

            up1 = nn.Upsample(scale_factor=2, mode='nearest')
            conv1 = nn.Conv3d(inner_nc * 2, inner_nc * 2, kernel_size=3, padding=1)
            
            upnorm1 = norm_layer(inner_nc * 2, affine=True)
            upconv2 = nn.Conv3d(inner_nc * 2, outer_nc,
                                    kernel_size=3, stride=1,
                                    padding=1)
            down = [downconv1, downnorm1, downrelu1, downconv2]
            up = [uprelu1, up1, conv1, upnorm1, uprelu2, upconv2, nn.Tanh()]
            model = down + [submodule] + up
            upnorm2 = None
   
        elif innermost:
 
            up1 = nn.Upsample(scale_factor=2, mode='nearest')
            conv1 = nn.Conv3d(inner_nc, outer_nc, kernel_size=3, padding=1)

            down = [downrelu1, downconv1, downnorm1, downrelu2, downconv2]
            up = [uprelu1, up1, conv1, upnorm1, uprelu2, upconv2, upnorm2]
            model = down + up
        else:

            up1 = nn.Upsample(scale_factor=2, mode='nearest')
            conv1 = nn.Conv3d(inner_nc * 2, outer_nc, kernel_size=3, padding=1)
            down = [downrelu1, downconv1, downnorm1, downrelu2, downconv2, downnorm2]
            up = [uprelu1, up1, conv1, upnorm1, uprelu2, upconv2, upnorm2]
       
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        if self.outermost:
            # print('x', x.shape)
            return self.model(x)
        else:
            # print('x', x.shape)
            y = self.model(x)
            if x.shape != y.shape:
                y = F.interpolate(y, size=x.shape[-3:])
            # print('y', y.shape)
            return torch.cat([x, y], 1)
            
            # return torch.cat([x, self.model(x)], 1)
class Anisotropy_Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Anisotropy_Upsample, self).__init__()
        self.name = type(self).__name__
        self.size = size
        self.scale_factor = scale_factor if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners
    def forward(self, input):
        # batch_size = input.shape[0]
        # channel = input.shape[1]
        # input = input.view(batch_size*channel, input.shape[2],input.shape[3],input.shape[4])
        # input = F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)
        # input = input.view(batch_size, channel, input.shape[1],input.shape[2],input.shape[3])
        # return input
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info

class UnetSkipConnection_nonzdownsaple_Block(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm3d, use_dropout=False):
        super(UnetSkipConnection_nonzdownsaple_Block, self).__init__()  
        self.outermost = outermost
        self.submodule = submodule

        downconv1 = nn.Conv3d(outer_nc, inner_nc, kernel_size=(1,3,3),
                             stride=(1,2,2), padding=(0, 1, 1))
        downrelu1 = nn.ReLU(True)

        downconv2 = nn.Conv3d(inner_nc, inner_nc, kernel_size=3, padding=1)
        downrelu2 = nn.ReLU(True)

        downnorm1 = norm_layer(inner_nc, affine=True)
        downnorm2 = norm_layer(inner_nc, affine=True)
        uprelu1 = nn.ReLU(True)
        upnorm1 = norm_layer(outer_nc, affine=True)
        upconv2 = nn.Conv3d(outer_nc, outer_nc,
                                    kernel_size=(1,3,3), stride=1, padding=(0, 1, 1))
        upnorm2 = norm_layer(outer_nc, affine=True)
        uprelu2 = nn.ReLU(True)
        
        if outermost:

            up1 = Anisotropy_Upsample(scale_factor=(1.0,2.0,2.0), mode='nearest')
            
            conv1 = nn.Conv3d(inner_nc * 2, inner_nc * 2, kernel_size=(1,3,3), stride=1, padding=(0, 1, 1))
            
            upnorm1 = norm_layer(inner_nc * 2, affine=True)
            upconv2 = nn.Conv3d(inner_nc * 2, outer_nc,
                                    kernel_size=(1,3,3), stride=1, padding=(0, 1, 1))
            down = [downconv1, downnorm1, downrelu1, downconv2]
            up = [uprelu1, up1, conv1, upnorm1, uprelu2, upconv2, nn.Tanh()]
            model = down + [submodule] + up
            upnorm2 = None
   
        elif innermost:
 
            up1 = Anisotropy_Upsample(scale_factor=(1.0,2.0,2.0), mode='nearest')
            conv1 = nn.Conv3d(inner_nc, outer_nc, kernel_size=(1,3,3), stride=1, padding=(0, 1, 1))

            down = [downrelu1, downconv1, downnorm1, downrelu2, downconv2]
            up = [uprelu1, up1, conv1, upnorm1, uprelu2, upconv2, upnorm2]
            model = down + up
        else:

            up1 = Anisotropy_Upsample(scale_factor=(1.0,2.0,2.0), mode='nearest')
            conv1 = nn.Conv3d(inner_nc * 2, outer_nc, kernel_size=(1,3,3), stride=1, padding=(0, 1, 1))
            down = [downrelu1, downconv1, downnorm1, downrelu2, downconv2, downnorm2]
            up = [uprelu1, up1, conv1, upnorm1, uprelu2, upconv2, upnorm2]
       
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        if self.outermost:
            # print('x', x.shape)
            return self.model(x)
        else:
            # print('x', x.shape)
            y = self.model(x)
            if x.shape != y.shape:
                y = F.interpolate(y, size=x.shape[-3:])
            # print('y', y.shape)
            return torch.cat([x, y], 1)
            # return torch.cat([x, self.model(x)], 1)

class UnetSkipConnection_133_Block(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm3d, use_dropout=False):
        super(UnetSkipConnection_133_Block, self).__init__()  
        self.outermost = outermost
        self.submodule = submodule

        downconv1 = nn.Conv3d(outer_nc, inner_nc, kernel_size=(1,3,3),
                             stride=(1,2,2), padding=(0, 1, 1))
        downrelu1 = nn.ReLU(True)

        downconv2 = nn.Conv3d(inner_nc, inner_nc, kernel_size=(1,3,3), stride=1, padding=(0, 1, 1))
        downrelu2 = nn.ReLU(True)

        downnorm1 = norm_layer(inner_nc, affine=True)
        downnorm2 = norm_layer(inner_nc, affine=True)
        uprelu1 = nn.ReLU(True)
        upnorm1 = norm_layer(outer_nc, affine=True)
        upconv2 = nn.Conv3d(outer_nc, outer_nc,
                                    kernel_size=(1,3,3), stride=1, padding=(0, 1, 1))
        upnorm2 = norm_layer(outer_nc, affine=True)
        uprelu2 = nn.ReLU(True)
        
        if outermost:

            up1 = Anisotropy_Upsample(scale_factor=(1.0,2.0,2.0), mode='nearest')
            
            conv1 = nn.Conv3d(inner_nc * 2, inner_nc * 2, kernel_size=(1,3,3), stride=1, padding=(0, 1, 1))
            
            upnorm1 = norm_layer(inner_nc * 2, affine=True)
            upconv2 = nn.Conv3d(inner_nc * 2, outer_nc,
                                    kernel_size=(1,3,3), stride=1, padding=(0, 1, 1))
            down = [downconv1, downnorm1, downrelu1, downconv2]
            up = [uprelu1, up1, conv1, upnorm1, uprelu2, upconv2, nn.Tanh()]
            model = down + [submodule] + up
            upnorm2 = None
   
        elif innermost:
 
            up1 = Anisotropy_Upsample(scale_factor=(1.0,2.0,2.0), mode='nearest')
            conv1 = nn.Conv3d(inner_nc, outer_nc, kernel_size=(1,3,3), stride=1, padding=(0, 1, 1))

            down = [downrelu1, downconv1, downnorm1, downrelu2, downconv2]
            up = [uprelu1, up1, conv1, upnorm1, uprelu2, upconv2, upnorm2]
            model = down + up
        else:

            up1 = Anisotropy_Upsample(scale_factor=(1.0,2.0,2.0), mode='nearest')
            conv1 = nn.Conv3d(inner_nc * 2, outer_nc, kernel_size=(1,3,3), stride=1, padding=(0, 1, 1))
            down = [downrelu1, downconv1, downnorm1, downrelu2, downconv2, downnorm2]
            up = [uprelu1, up1, conv1, upnorm1, uprelu2, upconv2, upnorm2]
       
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        if self.outermost:
            # print('x', x.shape)
            return self.model(x)
        else:
            # print('x', x.shape)
            y = self.model(x)
            if x.shape != y.shape:
                y = F.interpolate(y, size=x.shape[-3:])
            # print('y', y.shape)
            return torch.cat([x, y], 1)
            # return torch.cat([x, self.model(x)], 1)

class UnetGenerator_default(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator_default, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock_default(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        # for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
        #     unet_block = UnetSkipConnectionBlock_default(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock_default(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # unet_block = UnetSkipConnectionBlock_default(ngf, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock_default(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock_default(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock_default(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock_default(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock_default, self).__init__()
        self.outermost = outermost
        # self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            # downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=3,
            #                     stride=1, padding=1, bias=use_bias)
            # upconv = nn.Conv3d(input_nc, inner_nc, kernel_size=3,
            #                     stride=1, padding=1, bias=use_bias)
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                            kernel_size=3, stride=2,
                            padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            # print('outermost',self.outermost)
            # print('innermost',self.innermost)
            # print('x',x.shape)
            return self.model(x)
        else:   # add skip connections
            print('outermost',self.outermost)
            # print('innermost',self.innermost)
            print('x',x.shape)
            print(self.model)
            y =  self.model(x)
            print('y', y.shape)
            return torch.cat([x, y], 1)
            # return torch.cat([x, self.model(x)], 1)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 3
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

import math
import torch.nn.functional as F
from models.block import *
from models.utils import *

class BilinearUp(nn.Module):
	def __init__(self, in_channels, out_channels, factor=(1,2,2)):
		super(BilinearUp, self).__init__()
		assert in_channels==out_channels
		self.groups = in_channels
		self.factor = factor
		self.kernel_size = [(2 * f) - (f % 2) for f in self.factor]
		self.padding = [int(math.ceil((f - 1) / 2.0)) for f in factor]
		self.init_weights()

	def init_weights(self):
		weight = torch.Tensor(self.groups, 1, *self.kernel_size)
		width = weight.size(-1)
		hight = weight.size(-2)
		assert width==hight
		f = float(math.ceil(width / 2.0))
		c = float(width - 1) / (2.0 * f)
		for w in range(width):
			for h in range(hight):
				weight[...,h,w] = (1 - abs(w/f - c)) * (1 - abs(h/f - c))
		self.register_buffer('weight', weight) # fixed

	def forward(self, x):
		return F.conv_transpose3d(x, self.weight, stride=self.factor, padding=self.padding, groups=self.groups)


class unet_3d(nn.Module):
    """Lightweight 3D U-net with residual blocks (based on [Lee2017]_ with modifications).

    .. [Lee2017] Lee, Kisuk, Jonathan Zung, Peter Li, Viren Jain, and 
        H. Sebastian Seung. "Superhuman accuracy on the SNEMI3D connectomics 
        challenge." arXiv preprint arXiv:1706.00120, 2017.
        
    Args:
        in_channel (int): number of input channels.
        out_channel (int): number of output channels.
        filters (list): number of filters at each u-net stage.
    """
    def __init__(self, in_channel=1, out_channel=3, filters=[28, 36, 48, 64, 80], pad_mode='rep', norm_mode='bn', act_mode='elu', 
                 do_embedding=False, head_depth=1, output_act='tanh'):
        super().__init__()

        self.depth = len(filters)-2
        self.do_embedding = do_embedding
        self.output_act = output_act # activation function for the output layer

        # encoding path
        if self.do_embedding: 
            num_out = filters[1]
            self.downE = nn.Sequential(                # anisotropic embedding
                conv3d_norm_act(in_planes=in_channel, out_planes=filters[0], 
                              kernel_size=(1,5,5), stride=1, padding=(0,2,2), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                # 2d residual module
                conv3d_norm_act(in_planes=filters[0], out_planes=filters[0], 
                              kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                residual_block_3d(filters[0], filters[0], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            )
        else:
            filters[0] = in_channel
            num_out = out_channel
        
        self.downC = nn.ModuleList(
            [nn.Sequential(
            conv3d_norm_act(in_planes=filters[x], out_planes=filters[x+1], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            residual_block_3d(filters[x+1], filters[x+1], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            ) for x in range(self.depth)])

        # pooling downsample
        # self.downS = nn.ModuleList([nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)) for x in range(self.depth+1)])
        self.downS = nn.ModuleList(  # 用conv3d 下采样
            [conv3d_norm_act(in_planes=filters[x], out_planes=filters[x], kernel_size=(1,3,3), stride=(1, 2, 2), padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            for x in range(self.depth+1)])

        # center block
        self.center = nn.Sequential(conv3d_norm_act(in_planes=filters[-2], out_planes=filters[-1], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            residual_block_3d(filters[-1], filters[-1], projection=False)
        )
        self.middle = nn.ModuleList(
            [nn.Sequential(
                conv3d_norm_act(in_planes=filters[x], out_planes=filters[x],
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            ) for x in range(self.depth+1)])
            
        self.upC = nn.ModuleList(
            [nn.Sequential(
                conv3d_norm_act(in_planes=filters[x+1], out_planes=filters[x+1], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                residual_block_3d(filters[x+1], filters[x+1], projection=False)
            ) for x in range(self.depth)])

        if self.do_embedding: 
            # decoding path
            self.upE = nn.Sequential(
                conv3d_norm_act(in_planes=filters[0], out_planes=filters[0], 
                              kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                residual_block_3d(filters[0], filters[0], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                conv3d_norm_act(in_planes=filters[0], out_planes=out_channel, 
                              kernel_size=(1,5,5), stride=1, padding=(0,2,2), pad_mode=pad_mode, norm_mode=norm_mode)
            )
            # conv + upsample
            self.upS = nn.ModuleList([nn.Sequential(
                            conv3d_norm_act(filters[x+1], filters[x], kernel_size=(1,1,1), padding=0, norm_mode=norm_mode, act_mode=act_mode),
                            BilinearUp(filters[x], filters[x], factor=(1,2,2))) for x in range(self.depth+1)])
        else:
            # new
            head_pred = [residual_block_3d(filters[1], filters[1], projection=False)
                                    for x in range(head_depth-1)] + \
                              [conv3d_norm_act(filters[1], out_channel, kernel_size=(1,1,1), padding=0, norm_mode=norm_mode)]
            self.upS = nn.ModuleList( [nn.Sequential(*head_pred)] + \
                                 [nn.Sequential(
                        conv3d_norm_act(filters[x+1], filters[x], kernel_size=(1,1,1), padding=0, norm_mode=norm_mode, act_mode=act_mode),
                                     BilinearUp(filters[x], filters[x], factor=(1, 2, 2))) for x in range(1,self.depth+1)])
            
            # old
            # self.upS = nn.ModuleList( [conv3d_norm_act(filters[1], out_channel, kernel_size=(1,1,1), padding=0, norm_mode=norm_mode)] + \
            #                      [nn.Sequential(
            #             conv3d_norm_act(filters[x+1], filters[x], kernel_size=(1,1,1), padding=0, norm_mode=norm_mode, act_mode=act_mode),
            #             nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)) for x in range(1,self.depth+1)])
            


        #initialization
        ortho_init(self)

    def forward(self, x):
        # encoding path
        if self.do_embedding:
            z = self.downE(x) # torch.Size([4, 1, 32, 256, 256])
            x = self.downS[0](z) # downsample

        down_u = [None] * (self.depth)
        for i in range(self.depth):
            down_u[i] = self.downC[i](x)
            x = self.downS[i+1](down_u[i]) # downsample

        x = self.center(x)

        # z torch.Size([2, 28, 32, 256, 256]

        # down_u[0] [2, 36, 32, 128, 128]
        # down_u[1] [2, 48, 32, 64, 64]
        # down_u[3] [2, 64, 32, 32, 32]
        # x [16, 16]
        if self.do_embedding: 
            z = self.middle[0](z)
        layer = []

        # layer1 = self.middle[1](down_u[0])
        # layer2 = self.middle[2](down_u[1])
        # layer3 = self.middle[3](down_u[2])
        # layer4 = down_u[3]

        for i in range(len(down_u)-1):
            layer.append(self.middle[i+1](down_u[i]))
            # print(i)

        # decoding path
        for i in range(self.depth-1,-1,-1):
            x = down_u[i] + self.upS[i+1](x)
            x = self.upC[i](x)

        if self.do_embedding: 
            x = z + self.upS[0](x)
            x = self.upE(x)
        else:
            x = self.upS[0](x)

        x = get_functional_act(self.output_act)(x)
        return x

class unet_3d_k3(nn.Module):
    """Lightweight 3D U-net with residual blocks (based on [Lee2017]_ with modifications).

    .. [Lee2017] Lee, Kisuk, Jonathan Zung, Peter Li, Viren Jain, and 
        H. Sebastian Seung. "Superhuman accuracy on the SNEMI3D connectomics 
        challenge." arXiv preprint arXiv:1706.00120, 2017.
        
    Args:
        in_channel (int): number of input channels.
        out_channel (int): number of output channels.
        filters (list): number of filters at each u-net stage.
    """
    def __init__(self, in_channel=1, out_channel=3, filters=[28, 36, 48, 64, 80], pad_mode='rep', norm_mode='bn', act_mode='elu', 
                 do_embedding=False, head_depth=1, output_act='tanh'):
        super().__init__()

        self.depth = len(filters)-2
        self.do_embedding = do_embedding
        self.output_act = output_act # activation function for the output layer

        # encoding path
        if self.do_embedding: 
            num_out = filters[1]
            self.downE = nn.Sequential(
                # anisotropic embedding
                conv3d_norm_act(in_planes=in_channel, out_planes=filters[0], 
                              kernel_size=(1,5,5), stride=1, padding=(0,2,2), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                # 2d residual module
                conv3d_norm_act(in_planes=filters[0], out_planes=filters[0], 
                              kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                residual_block_3d(filters[0], filters[0], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            )
        else:
            filters[0] = in_channel
            num_out = out_channel
        
        self.downC = nn.ModuleList(
            [nn.Sequential(
            conv3d_norm_act(in_planes=filters[x], out_planes=filters[x+1], 
                          kernel_size=(3,3,3), stride=1, padding=(1,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            residual_block_3d(filters[x+1], filters[x+1], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            ) for x in range(self.depth)])

        # pooling downsample
        # self.downS = nn.ModuleList([nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)) for x in range(self.depth+1)])
        self.downS = nn.ModuleList(  # 用conv3d 下采样
            [conv3d_norm_act(in_planes=filters[x], out_planes=filters[x], kernel_size=(3,3,3), stride=(2, 2, 2), padding=(1,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            for x in range(self.depth+1)])

        # center block
        self.center = nn.Sequential(conv3d_norm_act(in_planes=filters[-2], out_planes=filters[-1], 
                          kernel_size=(3,3,3), stride=1, padding=(1,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            residual_block_3d(filters[-1], filters[-1], projection=False)
        )
        self.middle = nn.ModuleList(
            [nn.Sequential(
                conv3d_norm_act(in_planes=filters[x], out_planes=filters[x],
                          kernel_size=(3,3,3), stride=1, padding=(1,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            ) for x in range(self.depth+1)])
            
        self.upC = nn.ModuleList(
            [nn.Sequential(
                conv3d_norm_act(in_planes=filters[x+1], out_planes=filters[x+1], 
                          kernel_size=(3,3,3), stride=1, padding=(1,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                residual_block_3d(filters[x+1], filters[x+1], projection=False)
            ) for x in range(self.depth)])

        if self.do_embedding: 
            # decoding path
            self.upE = nn.Sequential(
                conv3d_norm_act(in_planes=filters[0], out_planes=filters[0], 
                              kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                residual_block_3d(filters[0], filters[0], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                conv3d_norm_act(in_planes=filters[0], out_planes=out_channel, 
                              kernel_size=(1,5,5), stride=1, padding=(0,2,2), pad_mode=pad_mode, norm_mode=norm_mode)
            )
            # conv + upsample
            self.upS = nn.ModuleList([nn.Sequential(
                            conv3d_norm_act(filters[x+1], filters[x], kernel_size=(1,1,1), padding=0, norm_mode=norm_mode, act_mode=act_mode),
                            BilinearUp(filters[x], filters[x], factor=(2,2,2))) for x in range(self.depth+1)])
        else:
            # new
            head_pred = [residual_block_3d(filters[1], filters[1], projection=False)
                                    for x in range(head_depth-1)] + \
                              [conv3d_norm_act(filters[1], out_channel, kernel_size=(1,1,1), padding=0, norm_mode=norm_mode)]
            self.upS = nn.ModuleList( [nn.Sequential(*head_pred)] + \
                                 [nn.Sequential(
                        conv3d_norm_act(filters[x+1], filters[x], kernel_size=(1,1,1), padding=0, norm_mode=norm_mode, act_mode=act_mode),
                                     BilinearUp(filters[x], filters[x], factor=(2, 2, 2))) for x in range(1,self.depth+1)])
            
            # old
            # self.upS = nn.ModuleList( [conv3d_norm_act(filters[1], out_channel, kernel_size=(1,1,1), padding=0, norm_mode=norm_mode)] + \
            #                      [nn.Sequential(
            #             conv3d_norm_act(filters[x+1], filters[x], kernel_size=(1,1,1), padding=0, norm_mode=norm_mode, act_mode=act_mode),
            #             nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)) for x in range(1,self.depth+1)])
            


        #initialization
        ortho_init(self)

    def forward(self, x):
        # encoding path
        if self.do_embedding:
            z = self.downE(x) # torch.Size([4, 1, 32, 256, 256])
            x = self.downS[0](z) # downsample

        down_u = [None] * (self.depth)
        for i in range(self.depth):
            down_u[i] = self.downC[i](x)
            x = self.downS[i+1](down_u[i]) # downsample

        x = self.center(x)

        # z torch.Size([2, 28, 32, 256, 256]

        # down_u[0] [2, 36, 32, 128, 128]
        # down_u[1] [2, 48, 32, 64, 64]
        # down_u[3] [2, 64, 32, 32, 32]
        # x [16, 16]
        if self.do_embedding: 
            z = self.middle[0](z)
        layer = []

        # layer1 = self.middle[1](down_u[0])
        # layer2 = self.middle[2](down_u[1])
        # layer3 = self.middle[3](down_u[2])
        # layer4 = down_u[3]

        for i in range(len(down_u)-1):
            layer.append(self.middle[i+1](down_u[i]))
            # print(i)

        # decoding path
        for i in range(self.depth-1,-1,-1):
            x = down_u[i] + self.upS[i+1](x)
            x = self.upC[i](x)

        if self.do_embedding: 
            x = z + self.upS[0](x)
            x = self.upE(x)
        else:
            x = self.upS[0](x)

        x = get_functional_act(self.output_act)(x)
        return x