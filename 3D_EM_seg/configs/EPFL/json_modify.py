import json
import os

def modify_im_train():
    js_path = 'im_val_epfl.json'
    my_path = '/braindat/lab/qic/data/PDAM/EM_DATA/2d_epfl/test/img_pad'
    img_list = os.listdir(my_path)
    img_list.sort()
    with open(js_path, 'r') as fp:
        data = json.load(fp)
    data['image'] = []
    for i in range(len(img_list)):
        data['image'].append(os.path.join(my_path, img_list[i])) 

    with open(js_path, 'w') as fp:
        json.dump(data, fp)
    
def modify_mito_train():
    js_path = 'label_.json'
    my_path = '/data/MitoEM-R/'
    with open(js_path, 'r') as fp:
        data = json.load(fp)

    for i in range(len(data['image'])):
        x = data['image'][i]
        x = x.strip().split('/')
        data['image'][i] = my_path+'/'.join(x[-2:])

    with open(js_path, 'w') as fp:
        json.dump(data, fp)

if __name__ == '__main__':
    # modify_mito_train()
    modify_im_train()