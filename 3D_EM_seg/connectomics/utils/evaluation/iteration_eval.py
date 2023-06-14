"""
This script looks very complicated, but in fact most of them are default parameters.
There are some important parameters:
SYSTEM.NUM_GPUS
SYSTEM.NUM_CPUS
INFERENCE.INPUT_SIZE: Although the training size is only 32 × 256 × 256, we have empirically found that D=100 is better.
                      (Thanks to the fully convolutional network, the input size is variable)
INFERENCE.STRIDE
INFERENCE.PAD_SIZE
INFERENCE.AUG_NUM: 0 is faster
"""
import os
import subprocess
from connectomics.utils.evaluation.evaluate_epfl import eval_epfl
from connectomics.utils.evaluation.evaluate import eval_mito

def cal_infer_epfl(root_dir, model_dir, model_id, pre_dir, yaml_dir):
    print("start inference...")

    command = "python scripts/main.py --config-file\
                {}\
                --inference\
                --do_h5\
                --checkpoint\
                {}/{}/checkpoint_{:06d}.pth.tar\
                --opts\
                SYSTEM.ROOTDIR\
                {}\
                INFERENCE.SAMPLES_PER_BATCH\
                2\
                INFERENCE.INPUT_SIZE\
                [165,256,256]\
                INFERENCE.OUTPUT_SIZE\
                [165,256,256]\
                INFERENCE.STRIDE\
                [1,256,256]\
                INFERENCE.PAD_SIZE\
                [0,256,256]\
            ".format(yaml_dir, root_dir, model_dir, model_id, root_dir)  # [0,256,256] [0,0,0]

    # os.system(command)
    
    out = subprocess.run(command, shell=True)
    print("\n |-------------| \n", out, "\n |-------------| \n")
    print('inference is done')
    
    # command = "python connectomics/utils/evaluation/evaluate_epfl.py \
    #              -external_input \
    #              -gt_instance \
    #              /braindat/lab/qic/data/PDAM/EM_DATA/EPFL/raw_data/testing_instance_groundtruth.tif \
    #              -p_instance \
    #              {}/{}/{:06d}_out_165_256_256_aug_0_pad_0.tif \
    #              -o {}/{}".format(root_dir, pre_dir, model_id, root_dir, pre_dir)

    # out = subprocess.run(command, shell=True)
    # print("\n |-------------| \n", out, "\n |-------------| \n")
    gt_instance = '/braindat/lab/qic/data/PDAM/EM_DATA/EPFL/raw_data/testing_instance_groundtruth.tif'
    p_instance = '{}/{}/{:06d}_out_165_256_256_aug_4_pad_0.tif'.format(root_dir, pre_dir, model_id)
    output_txt = '{}/{}'.format(root_dir, pre_dir)
    score = eval_epfl(gt_instance, p_instance, output_txt)
    return score

def cal_infer_mitoH(root_dir, model_dir, model_id, pre_dir, yaml_dir):
    """
    If you have enough resources, you can use this function during training. 
    Confirm that this line is open. 
    https://github.com/Limingxing00/MitoEM2021-Challenge/blob/dddb388a4aab004fa577058b53c39266e304fc03/connectomics/engine/trainer.py#L423
    """

    command = "python scripts/main.py --config-file\
                {}\
                --inference\
                --do_h5\
                --checkpoint\
                {}/{}/checkpoint_{:06d}.pth.tar\
                --opts\
                SYSTEM.ROOTDIR\
                {}\
                SYSTEM.NUM_GPUS\
                2\
                SYSTEM.NUM_CPUS\
                12\
                DATASET.DATA_CHUNK_NUM\
                [1,1,1]\
                INFERENCE.SAMPLES_PER_BATCH\
                2\
                INFERENCE.INPUT_SIZE\
                [100,256,256]\
                INFERENCE.OUTPUT_SIZE\
                [100,256,256]\
                INFERENCE.STRIDE\
                [1,256,256]\
                INFERENCE.PAD_SIZE\
                [0,256,256]\
                INFERENCE.AUG_NUM\
                0\
            ".format(yaml_dir, root_dir, model_dir, model_id, root_dir)

    out = subprocess.run(command, shell=True)
    print(command, "\n |-------------| \n", out, "\n |-------------| \n")

    # command = "python connectomics/utils/evaluation/evaluate.py \
    #              -external_input \
    #              -gt \
    #              /braindat/lab/qic/data/PDAM/EM_DATA/Mito/raw_data/human_val_gt.h5 \
    #              -p \
    #              {}/{}/{:06d}_out_100_256_256_aug_0_pad_0.h5 \
    #          -o {}/{}/{:06d}".format(root_dir, pre_dir, model_id, root_dir, pre_dir, model_id)
    # out = subprocess.run(command, shell=True)
    # print(command, "\n |-------------| \n", out, "\n |-------------| \n")

    gt = '/braindat/lab/qic/data/PDAM/EM_DATA/Mito/raw_data/human_val_gt.h5'
    p = '{}/{}/{:06d}_out_100_256_256_aug_0_pad_0.h5'.format(root_dir, pre_dir, model_id)
    output_name = '{}/{}/{:06d}'.format(root_dir, pre_dir, model_id)
    score = eval_mito(gt, p, output_name)
    return score

def cal_infer_mitoR(root_dir, model_dir, model_id, pre_dir, yaml_dir):
    """
    If you have enough resources, you can use this function during training. 
    Confirm that this line is open. 
    https://github.com/Limingxing00/MitoEM2021-Challenge/blob/dddb388a4aab004fa577058b53c39266e304fc03/connectomics/engine/trainer.py#L423
    """

    command = "python scripts/main.py --config-file\
                {}\
                --inference\
                --do_h5\
                --checkpoint\
                {}/{}/checkpoint_{:06d}.pth.tar\
                --opts\
                SYSTEM.ROOTDIR\
                {}\
                SYSTEM.NUM_GPUS\
                2\
                SYSTEM.NUM_CPUS\
                12\
                DATASET.DATA_CHUNK_NUM\
                [1,1,1]\
                INFERENCE.SAMPLES_PER_BATCH\
                2\
                INFERENCE.INPUT_SIZE\
                [100,256,256]\
                INFERENCE.OUTPUT_SIZE\
                [100,256,256]\
                INFERENCE.STRIDE\
                [1,256,256]\
                INFERENCE.PAD_SIZE\
                [0,256,256]\
                INFERENCE.AUG_NUM\
                0\
            ".format(yaml_dir, root_dir, model_dir, model_id, root_dir)

    out = subprocess.run(command, shell=True)
    print(command, "\n |-------------| \n", out, "\n |-------------| \n")

    # command = "python connectomics/utils/evaluation/evaluate.py \
    #              -external_input \
    #              -gt \
    #              /braindat/lab/qic/data/PDAM/EM_DATA/Mito/raw_data/rat_val_gt.h5 \
    #              -p \
    #              {}/{}/{:06d}_out_100_256_256_aug_0_pad_0.h5 \
    #          -o {}/{}/{:06d}".format(root_dir, pre_dir, model_id, root_dir, pre_dir, model_id)
    # out = subprocess.run(command, shell=True)
    # print(command, "\n |-------------| \n", out, "\n |-------------| \n")

    gt = '/braindat/lab/qic/data/PDAM/EM_DATA/Mito/raw_data/rat_val_gt.h5'
    p = '{}/{}/{:06d}_out_100_256_256_aug_0_pad_0.h5'.format(root_dir, pre_dir, model_id)
    output_name = '{}/{}/{:06d}'.format(root_dir, pre_dir, model_id)
    score = eval_mito(gt, p, output_name)
    return score

if __name__=="__main__":
    """
    Please note to change the gt file!
    My gt file is in:
    /braindat/lab/limx/MitoEM2021/MitoEM-H/MitoEM-H/human_val_gt.h5
    """
    # change the "start_epoch" and "end_epoch"  to infer "root_dir/model"
    start_epoch, end_epoch = 297000, 297000
    step_epoch = 2500
    model_id = range(start_epoch, end_epoch+step_epoch, step_epoch)

    # root_dir = "/braindat/lab/limx/MitoEM2021/CODE/HUMAN/rsunet_retrain_297000_v2/"
    root_dir = "/braindat/lab/limx/MitoEM2021/CODE/HUMAN/rsunet_retrain_297000_v2/"

    # validation stage: output h5
    # test stage: don't output h5
    for i in range(len(model_id)):  
        command = "python scripts/main.py --config-file\
                configs/MitoEM/MitoEM-R-BC.yaml\
                --inference\
                --do_h5\
                --checkpoint\
                {}outputs/dataset_output/checkpoint_{:06d}.pth.tar\
                --opts\
                SYSTEM.ROOTDIR\
                {}\
                SYSTEM.NUM_GPUS\
                2\
                SYSTEM.NUM_CPUS\
                8\
                DATASET.DATA_CHUNK_NUM\
                [1,1,1]\
                INFERENCE.SAMPLES_PER_BATCH\
                4\
                INFERENCE.INPUT_SIZE\
                [100,256,256]\
                INFERENCE.OUTPUT_SIZE\
                [100,256,256]\
                INFERENCE.STRIDE\
                [1,128,128]\
                INFERENCE.PAD_SIZE\
                [0,128,128]\
                INFERENCE.AUG_NUM\
                0\
                ".format(root_dir, model_id[i], root_dir)

        out = subprocess.run(command, shell=True)
        print(command, "\n |-------------| \n", out, "\n |-------------| \n")


        command = "python {}connectomics/utils/evaluation/evaluate.py \
             -gt \
             /braindat/lab/limx/MitoEM2021/MitoEM-H/MitoEM-H/human_val_gt.h5 \
             -p \
             {}outputs/inference_output/{:06d}_out_100_256_256_aug_0_pad_0.h5 \
                 -o {}/{:06d}".format(root_dir, root_dir, model_id[i], root_dir, model_id[i])

        out = subprocess.run(command, shell=True)
        print(command, "\n |-------------| \n", out, "\n |-------------| \n")
