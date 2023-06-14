from imageio.core.functions import volwrite
import numpy as np
import os
from numpy.lib.npyio import save
import skimage.io as io
import h5py
import imageio as io
import glob
def padding_zero(img_dir, label_dir, save_dir):
    img_list = os.listdir(img_dir)
    img_list.sort()
    img_list = [os.path.join(img_dir, x) for x in img_list]
    label_list = os.listdir(label_dir)
    label_list.sort()
    label_list = [os.path.join(label_dir, x) for x in label_list]
    for i in range(len(img_list)):
        img_pad = np.zeros((1024, 1024))
        label_pad = np.zeros((1024, 1024))
        img = io.imread(img_list[i])
        label = io.imread(label_list[i])
        h, w = img.shape
        img_pad[:h,:w] = img
        label_pad[:h,:w] = label
        img_save = os.path.join(save_dir, 'img_pad', str(i).zfill(3)+'.png')
        label_save = os.path.join(save_dir, 'label_pad', str(i).zfill(3)+'.png')
        io.imsave(img_save, img_pad)
        io.imsave(label_save, label_pad)
        print(img_save)


def readh5(filename, dataset=''):
    fid = h5py.File(filename,'r')
    print('fid',fid)
    if dataset=='':
        dataset = list(fid)[0]
    return np.array(fid[dataset])

def writeh5(filename, dtarray, dataset='main'):
    fid=h5py.File(filename,'w')
    if isinstance(dataset, (list,)):
        for i,dd in enumerate(dataset):
            ds = fid.create_dataset(dd, dtarray[i].shape, compression="gzip", dtype=dtarray[i].dtype)
            ds[:] = dtarray[i]
    else:
        ds = fid.create_dataset(dataset, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
        ds[:] = dtarray
    fid.close()

def readimgs(filename):
    filelist = sorted(glob.glob(filename))
    num_imgs = len(filelist)

    # decide numpy array shape:
    img = io.imread(filelist[0])
    data = np.zeros((num_imgs, img.shape[0], img.shape[1]), dtype=np.uint8)
    data[0] = img
    # load all images
    if num_imgs > 1:
        for i in range(1, num_imgs):
            data[i] = io.imread(filelist[i])
    return data

def h5_tif(volume_path):
    volume = readh5(volume_path).astype('uint8')
    basename = os.path.basename(volume_path).split('.')[0]
    save_path = basename+'.tif'
    volwrite(save_path, volume) 


if __name__ == '__main__':
    # img_dir = '/braindat/lab/qic/data/PDAM/EM_DATA/2d_epfl/test/img'
    # label_dir = '/braindat/lab/qic/data/PDAM/EM_DATA/2d_epfl/test/label'
    # save_dir = '/braindat/lab/qic/data/PDAM/EM_DATA/2d_epfl/test'
    # padding_zero(img_dir, label_dir, save_dir)

    # h5_path = '/braindat/lab/limx/MitoEM2021/MitoEM-H/MitoEM-H/human_val_gt.h5'
    # h5_path = '/braindat/lab/limx/MitoEM2021/CODE/HUMAN/rsunet_retrain_297000_v2/outputs/inference_output/297000_out_100_256_256_aug_0_pad_0.h5'
    # h5_path = 'epfl_val_gt.h5'
    # h5_file = readh5(h5_path)
    # print(h5_file.shape)

    # out_name = 'epfl_test_gt.h5'
    # gt_test_path = '/braindat/lab/qic/data/PDAM/EM_DATA/EPFL/raw_data/testing_groundtruth.tif'
    # gt_test = io.volread(gt_test_path)
    # writeh5(out_name, gt_test)

    # label_dir = '/braindat/lab/qic/data/PDAM/EM_DATA/2d_epfl/test/label_pad/*.png'
    # gt_test = readimgs(label_dir)
    # print(gt_test.shape)
    # writeh5(out_name, gt_test)

    # h5_tif('/braindat/lab/qic/data/PDAM/EM_DATA/EPFL/crop_3d_z32/2k/outputs/inference_output/001000_out_165_256_256_aug_0_pad_0.h5')
    # h5_tif('/braindat/lab/qic/data/PDAM/EM_DATA/EPFL/crop_3d_z32/2k/outputs/inference_output/040000_out_165_256_256_aug_0_pad_0.h5')
    h5_tif('/braindat/lab/qic/data/PDAM/EM_DATA/Mito/raw_data/human_val_gt.h5')



    
