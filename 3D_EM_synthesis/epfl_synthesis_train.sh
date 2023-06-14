set -ex

python train.py --gpu_ids 0 --netG unet_256 --display_port 8051 --dataroot /braindat/lab/qic/data/PDAM/EM_DATA/EPFL/raw_data --name epfl3d_synt --norm batch --model pix2pix --direction AtoB --lambda_L1 100 --dataset_mode alignedonline --pool_size 0 --display_freq 50 --batch_size 1 --n_epochs 10 --n_epochs_decay 10 --input_nc 1 --output_nc 1
