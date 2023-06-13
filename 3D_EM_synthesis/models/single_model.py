import torch
from .base_model import BaseModel
from . import networks
from torchsummary import summary
import random

class SingleModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)    
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            if self.opt.patchD:
                self.model_names = ['G', 'D', 'D_P']
                self.loss_names = ['G_GAN', 'G_L1', 'D', 'D_P', 'G_A']
            else:
                self.loss_names = ['G_GAN', 'G_L1', 'D']
                self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # print(self.netG)
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            print(self.netD)
            # summary(self.netD, (2, 64, 256, 256))
            if self.opt.patchD:
                # self.netD_P = networks.define_D(opt.input_nc, opt.ndf,
                #                             opt.which_model_netD,
                #                             opt.n_layers_patchD, opt.norm, use_sigmoid, self.gpu_ids, True)

                self.netD_P = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD_P,
                                          opt.n_layers_patchD, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            if opt.use_wgan:
                self.criterionGAN = networks.DiscLossWGANGP().to(self.device)
            else:
                self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            if self.opt.patchD:
                self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D_P)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.noise > 0:
            self.noise = torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise/255.)
            self.real_A = self.real_A + self.noise
        self.fake_B = self.netG(self.real_A)  # G(A)
        
        if self.opt.patchD:
            d = self.real_A.size(2)
            h = self.real_A.size(4)
            w = self.real_A.size(3)
            d_offset = random.randint(0, max(0, d - self.opt.patchSize[0] - 1))
            h_offset = random.randint(0, max(0, h - self.opt.patchSize[1] - 1))
            w_offset = random.randint(0, max(0, w - self.opt.patchSize[2] - 1))
            self.fake_patch = self.fake_B[:,:, d_offset:d_offset + self.opt.patchSize[0], h_offset:h_offset + self.opt.patchSize[1], w_offset:w_offset + self.opt.patchSize[2]]
            self.real_patch = self.real_B[:,:, d_offset:d_offset + self.opt.patchSize[0], h_offset:h_offset + self.opt.patchSize[1], w_offset:w_offset + self.opt.patchSize[2]]
            self.input_patch = self.real_A[:,:, d_offset:d_offset + self.opt.patchSize[0], h_offset:h_offset + self.opt.patchSize[1], w_offset:w_offset + self.opt.patchSize[2]]
        if self.opt.patchD_3 > 0:
            self.fake_patch_1 = []
            self.real_patch_1 = []
            self.input_patch_1 = []
            for i in range(self.opt.patchD_3):
                self.fake_patch_1.append(self.fake_B[:,:, d_offset:d_offset + self.opt.patchSize[0], h_offset:h_offset + self.opt.patchSize[1], w_offset:w_offset + self.opt.patchSize[2]])
                self.real_patch_1.append(self.real_B[:,:, d_offset:d_offset + self.opt.patchSize[0], h_offset:h_offset + self.opt.patchSize[1], w_offset:w_offset + self.opt.patchSize[2]])
                self.input_patch_1.append(self.real_A[:,:, d_offset:d_offset + self.opt.patchSize[0], h_offset:h_offset + self.opt.patchSize[1], w_offset:w_offset + self.opt.patchSize[2]])
                    
    def backward_D_basic(self, netD, real_input, real_patch, fake_patch, use_ragan):
        fake_AB = torch.cat((real_input, fake_patch), 1) 
        pred_fake = self.netD(fake_AB.detach())

        real_AB = torch.cat((real_input, real_patch), 1)
        pred_real = self.netD(real_AB)
    
        if self.opt.use_wgan:
            loss_D_real = pred_real.mean()
            loss_D_fake = pred_fake.mean()
            loss_D = loss_D_fake - loss_D_real + self.criterionGAN.calc_gradient_penalty(netD, 
                                                pred_real, pred_fake)
        elif self.opt.use_ragan and use_ragan:
            loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), True) +
                                      self.criterionGAN(pred_fake - torch.mean(pred_real), False)) / 2
        else:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        # loss_D.backward()
        return loss_D
        

    def backward_D(self):
        self.loss_D = self.backward_D_basic(self.netD, self.real_A, self.real_B, self.fake_B, True)
        self.loss_D.backward()


    def backward_D_P(self):
        if self.opt.hybrid_loss:
            loss_D_P = self.backward_D_basic(self.netD_P, self.input_patch, self.real_patch, self.fake_patch, False)
            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P += self.backward_D_basic(self.netD_P, self.input_patch_1[i], self.real_patch_1[i], self.fake_patch_1[i], False)
                self.loss_D_P = loss_D_P/float(self.opt.patchD_3 + 1)
            else:
                self.loss_D_P = loss_D_P
        else:
            loss_D_P = self.backward_D_basic(self.netD_P, self.input_patch, self.real_patch, self.fake_patch, True)
            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P += self.backward_D_basic(self.netD_P, self.input_patch_1[i], self.real_patch_1[i], self.fake_patch_1[i], True)
                self.loss_D_P = loss_D_P/float(self.opt.patchD_3 + 1)
            else:
                self.loss_D_P = loss_D_P
        self.loss_D_P.backward()


    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_GAN
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients

        self.loss_G_A = 0
        if self.opt.patchD:
            fake_AB_patch = torch.cat((self.input_patch, self.fake_patch), 1)
            pred_fake_patch = self.netD_P(fake_AB_patch)
            if self.opt.hybrid_loss:
                self.loss_G_A = self.criterionGAN(pred_fake_patch, True)
            else:
                real_AB_patch = torch.cat((self.input_patch, self.real_patch), 1)
                pred_real_patch = self.netD_P(real_AB_patch)
                
                self.loss_G_A = (self.criterionGAN(pred_real_patch - torch.mean(pred_fake_patch), False) +
                                      self.criterionGAN(pred_fake_patch - torch.mean(pred_real_patch), True)) / 2
        if self.opt.patchD_3 > 0:   
            for i in range(self.opt.patchD_3):
                fake_AB_patch_1 = torch.cat((self.input_patch_1[i], self.fake_patch_1[i]), 1)
                pred_fake_patch_1 = self.netD_P(fake_AB_patch_1)
                if self.opt.hybrid_loss:
                    self.loss_G_A += self.criterionGAN(pred_fake_patch_1, True)
                else:
                    real_AB_patch_1 = torch.cat((self.input_patch_1[i], self.real_patch_1[i]), 1)
                    pred_real_patch_1 = self.netD_P(real_AB_patch_1)
                    
                    self.loss_G_A += (self.criterionGAN(pred_real_patch_1 - torch.mean(pred_fake_patch_1), False) +
                                        self.criterionGAN(pred_fake_patch_1 - torch.mean(pred_real_patch_1), True)) / 2

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_A/float(self.opt.patchD_3 + 1)
        self.loss_G.backward()



    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.set_requires_grad(self.netD_P, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        if self.opt.patchD:
            self.set_requires_grad(self.netD_P, True)
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        if not self.opt.patchD:
            self.optimizer_D.step()
        else:
            self.optimizer_D_P.zero_grad()
            self.backward_D_P()
            self.optimizer_D.step()
            self.optimizer_D_P.step()
