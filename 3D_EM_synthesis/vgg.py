from operator import mod
import os
import numpy as np
 
import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable 
import torch.cuda
import torchvision.transforms as transforms
import skimage.io as io
from skimage import transform
from PIL import Image
import visdom

TARGET_IMG_SIZE = 224
img_to_tensor = transforms.ToTensor()

def model_vis():
    # model = models.vgg16(pretrained=True)
    model=models.vgg16(pretrained=True).features[:2]
    model=model.eval()	# 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
    model.cuda()
    print(model)
    # feature = torch.nn.Sequential(*list(model.children())[:])
    # print(feature)

    # 下面是输出内容
    print(model._modules.keys())

def make_model():
    model_list = [1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29]
    model_list = [i+1 for i in model_list]
    models_list = []
    for i in model_list:
        model=models.vgg16(pretrained=True).features[:i]	# 其实就是定位到第28层，对照着上面的key看就可以理解
        model=model.eval()	# 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
        model.cuda()	# 将模型从CPU发送到GPU,如果没有GPU则删除该行
        models_list.append(model)
    return models_list
    
#特征提取
def extract_feature(model,imgpath):
    model.eval()		# 必须要有，不然会影响特征提取结果
    
    img=Image.open(imgpath)		# 读取图片
    img=img.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    tensor = img_to_tensor(img)	# 将图片转化成tensor
    tensor = tensor.repeat(3, 1, 1)   # (3, 224, 224)
    tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)
    print(tensor.shape)
    
    tensor=tensor.cuda()	# 如果只是在cpu上跑的话要将这行去掉
    result=model(Variable(tensor))
    result_npy=result.data.cpu().numpy()	# 保存的时候一定要记得转成cpu形式的，不然可能会出错
    result_mean = np.mean(result_npy, axis=0)
    return result_mean[0]	# 返回的矩阵shape是[1, 512, 14, 14]，这么做是为了让shape变回[512, 14,14]
    
if __name__=="__main__":
    vis = visdom.Visdom(env=u'test1')

    models_list=make_model()
    print(len(models_list))
    models_name = ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu3_3', 'relu4_1', 'relu4_2', 'relu4_3', 'relu5_1', 'relu5_2', 'relu5_3']
    img_dir = '/braindat/lab/qic/data/PDAM/EM_DATA/2d_epfl/crop/train/img_2'
    save_dir = '/braindat/lab/qic/data/PDAM/EM_DATA/2d_epfl/crop/train/vgg_img_2'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    imgs = os.listdir(img_dir)
    imgs.sort()
    imgs_path = [os.path.join(img_dir, i) for i in imgs]
    for num, model in enumerate(models_list):
        # print(models_name[num],model) 
        for img_num, path in enumerate(imgs_path):
            tmp = extract_feature(model, path)
            tmp = transform.resize(tmp, (256, 256))
            # save_path = os.path.join(save_dir, models_name[num], imgs[num])
            save_path = os.path.join(save_dir, models_name[num] + '_' + imgs[img_num])
            vis.heatmap(tmp)
            print(f'save_path:{save_path}')
            # io.imsave(save_path, tmp)
            if img_num == 3:
                break
        if num == 7:
            break
    
    # model_vis()
            

    # print(tmp.shape)	# 打印出得到的tensor的shape
    # print(tmp)		# 打印出tensor的内容，其实可以换成保存tensor的语句，这里的话就留给读者自由发挥了

