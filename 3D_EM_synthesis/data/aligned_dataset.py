import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import sys
import imageio as io
import numpy as np
import torch
# from .augmentation import build_train_augmentor

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.dir_A = os.path.join(opt.dataroot, opt.phase, 'label')
        self.dir_B = os.path.join(opt.dataroot, opt.phase, 'volume')
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        # self.augmentor = build_train_augmentor(self.opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        
        A_path = self.A_paths[index]
        A = io.volread(A_path)
        B_path = self.B_paths[index]
        B = io.volread(B_path)

        A = np.expand_dims(A, 0)
        B = np.expand_dims(B, 0)
        A = torch.from_numpy(A).to(dtype=torch.float)
        B = torch.from_numpy(B).to(dtype=torch.float)

        # return {'A': A, 'B': B, 'C': C, 'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path}
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
