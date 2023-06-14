import json
import os
def modify_im_train():
    js_path = 'im_train_rat.json'
    my_path = '/braindat/lab/limx/MitoEM2021/MitoEM-R/MitoEM-R/im'
    with open(js_path, 'r') as fp:
        data = json.load(fp)

    for i in range(len(data['image'])):
        x = 'im'+str(i).zfill(4)+'.png'
        # x = 'im'+str(400+i).zfill(4)+'.png'
        data['image'][i] = os.path.join(my_path, x)
        # x = x.strip().split('/')
        # data['image'][i] = my_path+'/'.join(x[-2:])
        # x = data['image'][i]
        # x = x.strip().split('/')
        # data['image'][i] = my_path+'/'.join(x[-2:])

    with open(js_path, 'w') as fp:
        json.dump(data, fp)
    
def modify_mito_train():
    js_path = 'mito_train_rat.json'
    my_path = '/braindat/lab/limx/MitoEM2021/MitoEM-R/MitoEM-R/mito_train'
    with open(js_path, 'r') as fp:
        data = json.load(fp)

    for i in range(len(data['image'])):
        x = 'seg'+str(i).zfill(4)+'.tif'
        # x = 'im'+str(400+i).zfill(4)+'.png'
        data['image'][i] = os.path.join(my_path, x)
        # x = data['image'][i]
        # x = x.strip().split('/')
        # data['image'][i] = my_path+'/'.join(x[-2:])

    with open(js_path, 'w') as fp:
        json.dump(data, fp)

if __name__ == '__main__':
    modify_mito_train()
    # modify_im_train()