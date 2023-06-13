import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import sys
import imageio as io
import numpy as np
import torch
import random
# from .augmentation import build_train_augmentor

def crop_volume(data, sz, st=(0, 0, 0)):  # C*D*W*H, C=1
    st = np.array(st).astype(np.int32)
    return data[st[0]:st[0]+sz[0], st[1]:st[1]+sz[1], st[2]:st[2]+sz[2]]

def getSegType(mid):
    m_type = np.uint64
    if mid<2**8:
        m_type = np.uint8
    elif mid<2**16:
        m_type = np.uint16
    elif mid<2**32:
        m_type = np.uint32
    return m_type

def count_volume(data_sz, vol_sz, stride):
    return 1 + np.ceil((data_sz - vol_sz) / stride.astype(float)).astype(int)

def relabel(seg, do_type=False):
    # get the unique labels
    uid = np.unique(seg)
    # ignore all-background samples
    if len(uid)==1 and uid[0] == 0:
        return seg

    uid = uid[uid > 0]
    mid = int(uid.max()) + 1 # get the maximum label for the segment

    # create an array from original segment id to reduced id
    m_type = seg.dtype
    if do_type:
        m_type = getSegType(mid)
    mapping = np.zeros(mid, dtype=m_type)
    mapping[uid] = np.arange(1, len(uid) + 1, dtype=m_type)
    return mapping[seg]


class AlignedmitohDataset(BaseDataset):
    def __init__(self,opt,
                 sample_volume_size=(32, 256, 256),
                 sample_label_size=(32, 256, 256),
                 sample_stride=(1, 1, 1),
                 sample_invalid_thres=[0, 0],
                 augmentor=None, 
                 iter_num=80000,
                 # options for rejection sampling
                 reject_size_thres= -1,
                 reject_after_aug=False, 
                 reject_p= 0.95):

        BaseDataset.__init__(self, opt)
        # data format
        self.volume_path = os.path.join(opt.dataroot, opt.volume_name)
        self.label_path = os.path.join(opt.dataroot, opt.label_name)
        self.volume = [io.volread(self.volume_path)]
        self.label = [io.volread(self.label_path)]
        self.augmentor = augmentor  # data augmentation


        # rejection samping
        self.reject_size_thres =  opt.reject_size_thres
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
        vol_size = self.sample_volume_size # [34, 505, 505]
        # train mode
        # if elastic deformation: need different receptive field
        # position in the volume
        pos, out_volume, out_label = self._rejection_sampling(vol_size, 
                        size_thres=self.reject_size_thres, 
                        background=0, 
                        p=self.reject_p)
        """ 
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
                """
            
        out_volume = np.expand_dims(out_volume, 0)
        out_label = np.expand_dims(out_label, 0)
        A = out_label
        B = out_volume
        A = torch.from_numpy(A).to(dtype=torch.float)
        B = torch.from_numpy(B).to(dtype=torch.float)

        return {'A': A, 'B': B}

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
        did = self._index_to_dataset(random.randint(0,self.sample_num_a-1))
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
                        temp = (temp!=background).astype(int).sum()
                    else:
                        # restrict the foreground mask at the center region after data augmentation
                        z, y, x = self.augmentor.input_size
                        z_start = (self.sample_label_size[0] - z) // 2
                        y_start = (self.sample_label_size[1] - y) // 2
                        x_start = (self.sample_label_size[2] - x) // 2

                        temp = out_label.copy()
                        temp = temp[z_start:z_start+z, y_start:y_start+y, x_start:x_start+x]
                        temp = (temp!=background).astype(int).sum()
                else:
                    temp = (out_label!=background).astype(int).sum()

                # reject sampling
                if temp > size_thres*32*256*256:
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

        out_label = relabel(out_label.copy()).astype(np.float32)
        return pos, out_volume, out_label
    

