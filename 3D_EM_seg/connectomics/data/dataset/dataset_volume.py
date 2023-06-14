from __future__ import print_function, division
from traceback import print_tb
import numpy as np
import random
from typing import Optional, List
import torch
import torch.utils.data
from ..utils import count_volume, crop_volume, relabel, seg_to_targets, seg_to_weights, readvol, seg_widen_border
from scipy.ndimage import zoom
from skimage.segmentation import clear_border
import imageio as io

TARGET_OPT_TYPE = List[str]
WEIGHT_OPT_TYPE = List[List[str]]
from ..augmentation import Compose
AUGMENTOR_TYPE = Optional[Compose]

# 3d volume dataset class
class VolumeDataset(torch.utils.data.Dataset):
    """
    Dataset class for 3D image volumes.

    # BUG: parameter cannot be delivered here.
    
    Args:
        volume (list): list of image volumes.
        label (list): list of label volumes. Default: None
        sample_volume_size (tuple, int): model input size.
        sample_label_size (tuple, int): model output size.
        sample_stride (tuple, int): stride size for sampling.
        sample_invalid_thres (float): threshold for invalid regions.
        augmentor: data augmentor for training. Default: None
        target_opt (list): list the model targets generated from segmentation labels.
        weight_opt (list): list of options for generating pixel-wise weight masks.
        mode (str): training or inference mode.
        do_2d (bool): load 2d samples from 3d volumes. Default: False
        iter_num (int): total number of training iterations (-1 for inference). Default: -1
        reject_size_thres (int): threshold to decide if a sampled volumes contains foreground objects. Default: 0
        reject_after_aug (bool): decide whether to reject a sample after data augmentation. Default: False 
        reject_p (float): probability of rejecting non-foreground volumes.
    """
    def __init__(self,
                 volume, 
                 label=None,
                 sample_volume_size=(8, 64, 64),
                 sample_label_size=(8, 64, 64),
                 sample_stride=(1, 1, 1),
                 sample_invalid_thres=[0, 0],
                 augmentor=None, 
                 target_opt=['1'], 
                 weight_opt=[['1']],
                 mode='train',
                 do_2d=False,
                 iter_num=-1,
                 # options for rejection sampling
                 reject_size_thres= 0,
                 reject_after_aug=False, 
                 reject_p= 0.98,
                 use_label_smooth=False, # # BUG: parameter cannot be delivered here.
                 label_smooth=0.1):

        self.mode = mode
        self.do_2d = do_2d
        if self.do_2d:
            assert (sample_volume_size[0]==1) * (sample_label_size[0]==1)
        # for partially labeled data
        # m1 (no): sample chunks with over certain percentage
        #   = online version: rejection sampling can be slow
        #   = offline version: need to keep track of the mask
        # self.label_ratio = label_ratio
        # m2: make sure the center is labeled

        # data format
        self.volume = volume
        self.label = label
        self.augmentor = augmentor  # data augmentation
        # self.prob_map = self.calculate_3D_volume_prob_map(np.expand_dims(np.expand_dims(self.label[0], axis=-1).transpose((1,2,0,3)), axis=0))
        # io.volsave('prob_map.tif',self.prob_map)
        reject_after_aug = True

        self.target_opt = target_opt  # target opt
        self.weight_opt = weight_opt  # loss opt 

        # rejection samping
        self.reject_size_thres = reject_size_thres
        self.reject_after_aug = reject_after_aug
        self.reject_p = reject_p

        # dataset: channels, depths, rows, cols
        self.volume_size = [np.array(x.shape) for x in self.volume]  # volume size, could be multi-volume input [array([ 121, 2986, 2986])]
        print('volume_size',self.volume_size)  # volume_size [array([ 165, 1536, 1536])]  volume_size [array([ 100, 4608, 4608])]
        self.sample_volume_size = np.array(sample_volume_size).astype(int)  # model input size
        if self.label is not None: 
            self.sample_label_size = np.array(sample_label_size).astype(int)  # model label size
            self.label_vol_ratio = self.sample_label_size / self.sample_volume_size

        # compute number of samples for each dataset (multi-volume input)
        self.sample_stride = np.array(sample_stride, dtype=int)
       
        self.sample_size = [count_volume(self.volume_size[x], self.sample_volume_size, self.sample_stride)
                            for x in range(len(self.volume_size))]

        # total number of possible inputs for each volume
        self.sample_num = np.array([np.prod(x) for x in self.sample_size])

        # do partial label
        self.sample_invalid = [False]*len(self.sample_num)
        self.sample_invalid_thres = sample_invalid_thres
        if sample_invalid_thres[0] > 0:
            # [invalid_ratio, max_num_trial]
            if self.label is not None:
                self.sample_invalid_thres[0] = int(np.prod(self.sample_label_size) * sample_invalid_thres[0])
                for i in range(len(self.sample_num)):
                    seg_bad = np.array([-1]).astype(self.label[i].dtype)[0]
                    if np.any(self.label[i] == seg_bad):
                        print('dataset %d: needs mask for invalid region'%(i))
                        self.sample_invalid[i] = True
                        self.sample_num[i] = np.count_nonzero(seg != seg_id)

        self.sample_num_a = np.sum(self.sample_num)
        self.sample_num_c = np.cumsum([0] + list(self.sample_num))
        self.use_label_smooth = use_label_smooth
        self.label_smooth = label_smooth

        if mode=='test': # for test
            self.sample_size_vol = [np.array([np.prod(x[1:3]), x[2]]) for x in self.sample_size]
       
        # For relatively small volumes, the total number of samples can be generated is smaller
        # than the number of samples required for training (i.e., iteration * batch size). Thus
        # we let the __len__() of the dataset return the larger value among the two during training. 
        if iter_num < 0: # inference mode
            self.iter_num = self.sample_num_a
        else: # training mode
            self.iter_num = max(iter_num, self.sample_num_a)
        print('Total number of samples to be generated: ', self.iter_num)

    def __len__(self):  # number of possible position
        return self.iter_num

    def __getitem__(self, index):
        # orig input: keep uint format to save cpu memory
        # sample: need float32

        vol_size = self.sample_volume_size # [34, 505, 505]
        # print('index', index)
        if self.mode in ['train','valid']:
            # train mode
            # if elastic deformation: need different receptive field
            # position in the volume
            pos, out_volume, out_label = self._rejection_sampling(vol_size, 
                            size_thres=self.reject_size_thres, 
                            background=0, 
                            p=self.reject_p)

            # pos = [0, 0, 0, 0]
            # out_volume, out_label = self._random_3D_crop(np.expand_dims(self.volume[0], axis=-1).transpose((1,2,0,3)),np.expand_dims(self.label[0], axis=-1).transpose((1,2,0,3)),random_crop_size=(256,256,32),  val=False, vol_prob=self.prob_map[0])
            # out_volume = out_volume.squeeze().transpose(2,0,1)
            # out_label = out_label.squeeze().transpose(2,0,1)

            # print('out_volume',out_volume.shape)
            # print('out_label',out_label.shape)
            io.volsave('sample_volume.tif', out_volume)
            io.volsave('sample_label.tif', out_label)
            
            # augmentation
            if self.augmentor is not None:  # augmentation
                if np.array_equal(self.augmentor.sample_size, out_label.shape):
                    # for warping: cv2.remap require input to be float32
                    # make labels index smaller. o/w uint32 and float32 are not the same for some values
                    data = {'image':out_volume, 'label':out_label}
                    augmented = self.augmentor(data)
                    out_volume, out_label = augmented['image'], augmented['label']
                else: # the data is already augmented in the rejection sampling step
                    pass
                
            if self.do_2d:
                out_volume = np.squeeze(out_volume) 
                out_label = np.squeeze(out_label)
            out_volume = np.expand_dims(out_volume, 0)
            # output list
            out_target = seg_to_targets(out_label, self.target_opt) # TODO: 我们的目标
            out_weight = seg_to_weights(out_target, self.weight_opt)

            # pos [0, 40, 1060, 2164]
            # out_volume (1, 32, 256, 256)
            # out_target list len:2. out_target[0].shape (1, 32, 256, 256)
            # out_weight transfer to numpy (2, 1, 1, 32, 256, 256)
            if self.use_label_smooth:
                n_class = 2
                label_smooth = self.label_smooth
                for i in range(len(out_target)):
                    out_target[i][out_target[i] == 0] = label_smooth / (n_class - 1)
                    out_target[i][out_target[i] == 1] = 1 - label_smooth
                # print()
                # print("#----use label smooth: {} ----# \n".format(label_smooth))
            return pos, out_volume, out_target, out_weight

        elif self.mode == 'test':
            # test mode
            pos = self._get_pos_test(index)
            out_volume = (crop_volume(self.volume[pos[0]], vol_size, pos[1:])/255.0).astype(np.float32)
            if self.do_2d:
                out_volume = np.squeeze(out_volume) 
            return pos, np.expand_dims(out_volume,0)

    #######################################################
    # Position Calculator
    #######################################################

    def _index_to_dataset(self, index):
        return np.argmax(index < self.sample_num_c) - 1  # which dataset

    def _index_to_location(self, index, sz):
        # index -> z,y,x
        # sz: [y*x, x]
        pos = [0, 0, 0]
        pos[0] = np.floor(index/sz[0])
        pz_r = index % sz[0]
        pos[1] = int(np.floor(pz_r/sz[1]))
        pos[2] = pz_r % sz[1]
        return pos

    def _get_pos_test(self, index):
        pos = [0, 0, 0, 0]
        did = self._index_to_dataset(index)
        pos[0] = did
        index2 = index - self.sample_num_c[did]
        pos[1:] = self._index_to_location(index2, self.sample_size_vol[did])
        # if out-of-bound, tuck in
        for i in range(1, 4):
            if pos[i] != self.sample_size[pos[0]][i-1]-1:
                pos[i] = int(pos[i] * self.sample_stride[i-1])
            else:
                pos[i] = int(self.volume_size[pos[0]][i-1]-self.sample_volume_size[i-1])
        return pos

    def _get_pos_train(self, vol_size):
        # random: multithread
        # np.random: same seed
        pos = [0, 0, 0, 0]
        # pick a dataset
        # import pdb
        # pdb.set_trace()
        did = self._index_to_dataset(random.randint(0,self.sample_num_a-1))
        # print('did', did)
        pos[0] = did
        # pick a position
        tmp_size = count_volume(self.volume_size[did], vol_size, self.sample_stride)
        tmp_pos = [random.randint(0,tmp_size[x]-1) * self.sample_stride[x] for x in range(len(tmp_size))]
        if self.sample_invalid[did]:
            # make sure the ratio of valid is high
            seg_bad = np.array([-1]).astype(self.label[did].dtype)[0]
            num_trial = 0
            while num_trial<self.sample_invalid_thres[1]:
                out_label = crop_volume(self.label[pos[0]], vol_size, tmp_pos)
                if (out_label!=seg_bad).sum() >= self.sample_invalid_thres[0]:
                    break
                num_trial += 1
        pos[1:] = tmp_pos 
        return pos

    #######################################################
    # Volume Sampler
    #######################################################

    def _rejection_sampling(self, vol_size, size_thres=-1, background=0, p=0.9):
        while True:
            pos, out_volume, out_label = self._random_sampling(vol_size)
            if size_thres > 0:
                if self.augmentor is not None:
                    assert np.array_equal(self.augmentor.sample_size, self.sample_label_size)
                    if self.reject_after_aug:
                        # decide whether to reject the sample after data augmentation
                        data = {'image':out_volume, 'label':out_label}
                        augmented = self.augmentor(data)
                        out_volume, out_label = augmented['image'], augmented['label']

                        temp = out_label.copy().astype(int)
                        # temp = (temp!=background).astype(int).sum()
                        temp = (out_label>background).astype(int).sum()
                        temp2 = (out_label==background).astype(int).sum()
                    else:
                        # restrict the foreground mask at the center region after data augmentation
                        z, y, x = self.augmentor.input_size
                        z_start = (self.sample_label_size[0] - z) // 2
                        y_start = (self.sample_label_size[1] - y) // 2
                        x_start = (self.sample_label_size[2] - x) // 2

                        temp = out_label.copy()
                        temp = temp[z_start:z_start+z, y_start:y_start+y, x_start:x_start+x]
                        # temp = (temp!=background).astype(int).sum()
                        temp = (out_label>background).astype(int).sum()
                        temp2 = (out_label==background).astype(int).sum()
                else:
                    # temp = (out_label!=background).astype(int).sum()
                    temp = (out_label>background).astype(int).sum()
                    temp2 = (out_label==background).astype(int).sum()

                # reject sampling
                if temp > size_thres and temp2 <= 0:
                    break
                # elif random.random() > p:
                #     break

            else: # no rejection sampling for the foreground mask
                break

        return pos, out_volume, out_label

    def _random_sampling(self, vol_size):
        pos = self._get_pos_train(vol_size)
        out_volume = (crop_volume(self.volume[pos[0]], vol_size, pos[1:])/255.0).astype(np.float32)
        # position in the label 
        pos_l = np.round(pos[1:]*self.label_vol_ratio)
        out_label = crop_volume(self.label[pos[0]], self.sample_label_size, pos_l)

        # The option of generating valid masks has been deprecated.
        # if self.sample_invalid_thres[0]>0:
        #     seg_bad = np.array([-1]).astype(out_label.dtype)[0]
        #     out_mask = out_label!=seg_bad
        # else:
        #     out_mask = torch.ones((1),dtype=torch.uint8)

        out_label = relabel(out_label.copy()).astype(np.float32)
        return pos, out_volume, out_label

    def calculate_3D_volume_prob_map(self, Y, Y_path=None, w_foreground=0.94, w_background=0.06, save_dir=None):

        if Y is None and Y_path is None:
            raise ValueError("'Y' or 'Y_path' need to be provided")

        if w_foreground + w_background > 1:
            raise ValueError("'w_foreground' plus 'w_background' can not be greater " "than one")

        prob_map = np.copy(Y).astype(np.float32)
        l = prob_map.shape[0]
        channels = prob_map.shape[-1]
        v = np.max(prob_map)

        if isinstance(prob_map, list):
            first_shape = prob_map[0][0].shape
        else:
            first_shape = prob_map[0].shape

        print("Constructing the probability map . . .")
        maps = []
        diff_shape = False
        for i in range(l):
            if isinstance(prob_map, list):
                _map = prob_map[i][0].copy().astype(np.float32)
            else:
                _map = prob_map[i].copy().astype(np.float32)

            for k in range(channels):
                for j in range(_map.shape[2]):
                    # Remove artifacts connected to image border
                    _map[:,:,j,k] = clear_border(_map[:,:,j,k])
                foreground_pixels = (_map[:,:,:,k] == v).sum()
                background_pixels = (_map[:,:,:,k] == 0).sum()

                if foreground_pixels == 0:
                    _map[:,:,:,k][np.where(_map[:,:,:,k] == v)] = 0
                else:
                    _map[:,:,:,k][np.where(_map[:,:,:,k] == v)] = w_foreground/foreground_pixels
                if background_pixels == 0:
                    _map[:,:,:,k][np.where(_map[:,:,:,k] == 0)] = 0
                else:
                    _map[:,:,:,k][np.where(_map[:,:,:,k] == 0)] = w_background/background_pixels

                # Necessary to get all probs sum 1
                s = _map[:,:,:,k].sum()
                if s == 0:
                    t = 1
                    for x in _map[:,:,:,k].shape: t *=x
                    _map[:,:,:,k].fill(1/t)
                else:
                    _map[:,:,:,k] = _map[:,:,:,k]/_map[:,:,:,k].sum()

            if first_shape != _map.shape: diff_shape = True
            maps.append(_map)

        if not diff_shape:
            for i in range(len(maps)):
                maps[i] = np.expand_dims(maps[i], 0)
            maps = np.concatenate(maps)
            return maps

        
    def _random_3D_crop(self, vol, vol_mask, random_crop_size, val=False, vol_prob=None, weight_map=None, draw_prob_map_points=False):
        rows, cols, deep = vol.shape[0], vol.shape[1], vol.shape[2]
        dx, dy, dz = random_crop_size
        assert rows >= dx
        assert cols >= dy
        assert deep >= dz
        if val:
            x, y, z, ox, oy, oz = 0, 0, 0, 0, 0, 0
        else:
            if vol_prob is not None:
                prob = vol_prob.ravel()

                # Generate the random coordinates based on the distribution
                choices = np.prod(vol_prob.shape)
                index = np.random.choice(choices, size=1, p=prob)
                coordinates = np.unravel_index(index, shape=vol_prob.shape)
                x = int(coordinates[0])
                y = int(coordinates[1])
                z = int(coordinates[2])
                ox = int(coordinates[0])
                oy = int(coordinates[1])
                oz = int(coordinates[2])

                # Adjust the coordinates to be the origin of the crop and control to
                # not be out of the volume
                if x < int(random_crop_size[0]/2):
                    x = 0
                elif x > vol.shape[0] - int(random_crop_size[0]/2):
                    x = vol.shape[0] - random_crop_size[0]
                else:
                    x -= int(random_crop_size[0]/2)

                if y < int(random_crop_size[1]/2):
                    y = 0
                elif y > vol.shape[1] - int(random_crop_size[1]/2):
                    y = vol.shape[1] - random_crop_size[1]
                else:
                    y -= int(random_crop_size[1]/2)

                if z < int(random_crop_size[2]/2):
                    z = 0
                elif z > vol.shape[2] - int(random_crop_size[2]/2):
                    z = vol.shape[2] - random_crop_size[2]
                else:
                    z -= int(random_crop_size[2]/2)
            else:
                ox = 0
                oy = 0
                oz = 0
                x = np.random.randint(0, rows - dx + 1)
                y = np.random.randint(0, cols - dy + 1)
                z = np.random.randint(0, deep - dz + 1)

        if draw_prob_map_points:
            return vol[x:(x+dx), y:(y+dy), z:(z+dz), :], vol_mask[x:(x+dx), y:(y+dy), z:(z+dz), :], ox, oy, oz, x, y, z
        else:
            if weight_map is not None:
                return vol[x:(x+dx), y:(y+dy), z:(z+dz), :], vol_mask[x:(x+dx), y:(y+dy), z:(z+dz), :],\
                    weight_map[x:(x+dx), y:(y+dy), z:(z+dz), :]
            else:
                return vol[x:(x+dx), y:(y+dy), z:(z+dz), :], vol_mask[x:(x+dx), y:(y+dy), z:(z+dz), :]

class SingleVolumeDataset(torch.utils.data.Dataset):
    """
    Dataset class for 3D image volumes.

    # BUG: parameter cannot be delivered here.
    
    Args:
        volume (list): list of image volumes.
        label (list): list of label volumes. Default: None
        sample_volume_size (tuple, int): model input size.
        sample_label_size (tuple, int): model output size.
        sample_stride (tuple, int): stride size for sampling.
        sample_invalid_thres (float): threshold for invalid regions.
        augmentor: data augmentor for training. Default: None
        target_opt (list): list the model targets generated from segmentation labels.
        weight_opt (list): list of options for generating pixel-wise weight masks.
        mode (str): training or inference mode.
        do_2d (bool): load 2d samples from 3d volumes. Default: False
        iter_num (int): total number of training iterations (-1 for inference). Default: -1
        reject_size_thres (int): threshold to decide if a sampled volumes contains foreground objects. Default: 0
        reject_after_aug (bool): decide whether to reject a sample after data augmentation. Default: False 
        reject_p (float): probability of rejecting non-foreground volumes.
    """
    def __init__(self,
                 cfg,
                 volume_paths, 
                 label_paths=None,
                 augmentor=None, 
                 target_opt=['1'], 
                 weight_opt=[['1']],
                 mode='train',
                 do_2d=False,
                 iter_num=-1):

        self.mode = mode
        self.do_2d = do_2d
        self.cfg = cfg
        # data format
        self.volume_paths = volume_paths
        self.label_paths = label_paths
        self.augmentor = augmentor  # data augmentation

        self.target_opt = target_opt  # target opt
        self.weight_opt = weight_opt  # loss opt 



        # For relatively small volumes, the total number of samples can be generated is smaller
        # than the number of samples required for training (i.e., iteration * batch size). Thus
        # we let the __len__() of the dataset return the larger value among the two during training. 
        self.iter_num = iter_num
        print('Total number of iteration: ', self.iter_num)

    def __len__(self):  # number of possible position
        assert len(self.volume_paths) == len(self.label_paths)
        return len(self.volume_paths)
        

    def __getitem__(self, index):
        # orig input: keep uint format to save cpu memory
        # sample: need float32
        if self.mode in ['train','valid']:
            # train mode
            pos = [0, 0, 0, 0]

            out_volume, out_label =  self._get_input(index)
            
            # augmentation
            if self.augmentor is not None:  # augmentation
                if np.array_equal(self.augmentor.sample_size, out_label.shape):
                    # for warping: cv2.remap require input to be float32
                    # make labels index smaller. o/w uint32 and float32 are not the same for some values
                    data = {'image':out_volume, 'label':out_label}
                    augmented = self.augmentor(data)
                    out_volume, out_label = augmented['image'], augmented['label']
                else: # the data is already augmented in the rejection sampling step
                    pass
           
            if self.do_2d:
                out_volume = np.squeeze(out_volume) 
                out_label = np.squeeze(out_label)
            out_volume = np.expand_dims(out_volume, 0)
            # output list
            out_target = seg_to_targets(out_label, self.target_opt) # TODO: 我们的目标
            out_weight = seg_to_weights(out_target, self.weight_opt)

            # pos [0, 40, 1060, 2164]
            # out_volume (1, 32, 256, 256)
            # out_target list len:2. out_target[0].shape (1, 32, 256, 256)
            # out_weight transfer to numpy (2, 1, 1, 32, 256, 256)

            return pos, out_volume, out_target, out_weight

    def _get_input(self, index):

        volume = readvol(self.volume_paths[index])  # 读入图像
        # print(f"volume shape (original): {volume[i].shape}")
        # if i%100 == 0:
        #     print(i)

        if (np.array(self.cfg.DATASET.DATA_SCALE)!=1).any():
            volume = zoom(volume, self.cfg.DATASET.DATA_SCALE, order=1)
        # 如果scale==1，就要镜像padding一圈（我猜是因为有augmentation，所以预先padding）
        # PAD_SIZE: [0, 0, 0]
        volume = np.pad(volume, ((self.cfg.DATASET.PAD_SIZE[0],self.cfg.DATASET.PAD_SIZE[0]), 
                                    (self.cfg.DATASET.PAD_SIZE[1],self.cfg.DATASET.PAD_SIZE[1]), 
                                    (self.cfg.DATASET.PAD_SIZE[2],self.cfg.DATASET.PAD_SIZE[2])), 'reflect')
        # print(f"volume shape (after scaling and padding): {volume[i].shape}")

        if self.mode=='train':
            label = readvol(self.label_paths[index])
            if (np.array(self.cfg.DATASET.DATA_SCALE)!=1).any():
                label = zoom(label, self.cfg.DATASET.DATA_SCALE, order=0) 
            if self.cfg.DATASET.LABEL_EROSION!=0: # LABEL_EROSION 1
                label = seg_widen_border(label, self.cfg.DATASET.LABEL_EROSION)
            if self.cfg.DATASET.LABEL_BINARY and label.max()>1: # LABEL_BINARY: False
                label = label // 255
            if self.cfg.DATASET.LABEL_MAG !=0:  # LABEL_MAG 0
                label = (label/self.cfg.DATASET.LABEL_MAG).astype(np.float32)
            label = np.pad(label, ((self.cfg.DATASET.PAD_SIZE[0],self.cfg.DATASET.PAD_SIZE[0]), 
                                        (self.cfg.DATASET.PAD_SIZE[1],self.cfg.DATASET.PAD_SIZE[1]), 
                                        (self.cfg.DATASET.PAD_SIZE[2],self.cfg.DATASET.PAD_SIZE[2])), 'reflect')
                    
        return volume, label