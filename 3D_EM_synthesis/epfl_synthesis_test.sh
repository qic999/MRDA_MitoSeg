set -ex

python test.py --dataroot --netG unet_256 --dataroot /braindat/lab/qic/data/PDAM/EM_DATA/EPFL/raw_data --name epfl3d_synt --model pix2pix --direction AtoB
