import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.draw import rectangle_perimeter

import torch
import torchvision
from torch.autograd import Variable
from torchvision.ops import nms

from src.preprocess.HRCenterNet import HRCenterNet
from src.utils import BBOX_IMAGE_INPUT_SIZE, BBOX_IMAGE_OUTPUT_SIZE, NMS_SCORE_THRESHOLD, IOU_THRESHOLD, SAVE_IMAGE_SIZE

def nms_bbox(img_h, img_w, predict, nms_score, iou_threshold):
    output_size = BBOX_IMAGE_OUTPUT_SIZE

    bbox = list()
    score_list = list()
    
    heatmap=predict.data.cpu().numpy()[0, 0, ...]
    offset_y = predict.data.cpu().numpy()[0, 1, ...]
    offset_x = predict.data.cpu().numpy()[0, 2, ...]
    width_map = predict.data.cpu().numpy()[0, 3, ...]
    height_map = predict.data.cpu().numpy()[0, 4, ...]
    
    
    for j in np.where(heatmap.reshape(-1, 1) >= nms_score)[0]:

        row = j // output_size 
        col = j - row*output_size
        
        bias_x = offset_x[row, col] * (img_w / output_size)
        bias_y = offset_y[row, col] * (img_h / output_size)

        width = width_map[row, col] * output_size * (img_w / output_size)
        height = height_map[row, col] * output_size * (img_h / output_size)

        score_list.append(heatmap[row, col])

        row = row * (img_w / output_size) + bias_y
        col = col * (img_h / output_size) + bias_x

        top = row - width // 2
        left = col - height // 2
        bottom = row + width // 2
        right = col + height // 2

        start = (top, left)
        end = (bottom, right)

        bbox.append([top, left, bottom, right])
        
    _nms_index = torchvision.ops.nms(torch.FloatTensor(bbox), scores=torch.flatten(torch.FloatTensor(score_list)), iou_threshold=iou_threshold)
        
    return bbox, _nms_index


def load_bbox_model(weight_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    if weight_dir is not None:
        print("Load checkpoint from " + weight_dir)
        checkpoint = torch.load(weight_dir, map_location="cpu")   

    model = HRCenterNet()
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    return model, device

'''
param wright_dir: path to the trained model weight
param eval_image: PIL.Image object 
'''
def predict_bbox(eval_image, model, device, vis=False):
    input_size = BBOX_IMAGE_INPUT_SIZE
    
    test_tx = torchvision.transforms.Compose([
            torchvision.transforms.Resize((input_size, input_size)),
            torchvision.transforms.ToTensor(),
    ])

    image_tensor = test_tx(eval_image)
    image_tensor = image_tensor.unsqueeze_(0)
    inp = Variable(image_tensor)
    inp = inp.to(device, dtype=torch.float)
    predict = model(inp)
    
    img_h, img_w = eval_image.size
    bbox_all, _nms_index = nms_bbox(img_h, img_w, predict, nms_score=NMS_SCORE_THRESHOLD, iou_threshold=IOU_THRESHOLD)

    im_draw = np.asarray(torchvision.transforms.functional.resize(eval_image, (img_w, img_h))).copy()
    im_transform = Image.fromarray(im_draw.astype('uint8'), 'RGB')

    bbox_ret = []
    for k in range(len(_nms_index)):
    
        top, left, bottom, right = bbox_all[_nms_index[k]]
        
        start = (top, left)
        end = (bottom, right)
        
        rr, cc = rectangle_perimeter(start, end=end,shape=(eval_image.size[1], eval_image.size[0]))
        im_draw[rr, cc] = (255, 0, 0)

        bbox_ret.append((left, top, right, bottom))

    if vis:
        plt.imshow(im_draw)
        plt.show()

    return bbox_ret



def crop_and_save(load_dir, save_dir, weight_dir):
    model, device = load_bbox_model(weight_dir=weight_dir)

    for font in os.listdir(load_dir):
        font_path = os.path.join(load_dir, font)
        print("Checking directory: ", font_path)
        if os.path.isdir(font_path):
            for file in os.listdir(font_path):
                file_path = os.path.join(font_path, file)
                if os.path.isfile(file_path):
                    print("Found image: ", file_path)
                    image = Image.open(file_path).convert("RGB")
                    bbox = predict_bbox(model=model, device=device, eval_image=image)
                    for index, crop_tuple in enumerate(bbox):
                        image_crop = image.crop(crop_tuple)
                        image_crop = image_crop.resize((SAVE_IMAGE_SIZE, SAVE_IMAGE_SIZE))
                        save_subdir = "{}/{}".format(save_dir, font)
                        if not os.path.exists(save_subdir):
                            os.makedirs(save_subdir)
                        save_path = "{}/{}_{}".format(save_subdir, index, file)
                        image_crop.save(save_path)


if __name__ == "__main__":
    # param load_dir: raw image dataset following the below durectiry tree
    # param save_dir: where preprocessed image dataset will be saved to. The preprocessed 
    #                 dataset will be organized in a similar way as the raw image dataset
    # param weight_dir: where the weight for chinese characer segmentation net is saved. 
    #                 Default path is <project>/src/preprocess/HRCenterNet.pth.tar
    import os.path as osp
    load_dir = '/Users/garybluedemac/Desktop/advance_topic/project/project301/project301/data_raw'
    save_dir = '/Users/garybluedemac/Desktop/advance_topic/project/project301/project301/data'
    crop_and_save(load_dir=load_dir, 
                  save_dir=save_dir,
                  weight_dir=osp.join(osp.dirname(__file__), 'HRCenterNet.pth.tar'))

# please organize your raw image dataset in the following manner for crop_and_save for work properly:
# <directory tree>
# dataset
# -- caoshu
# ---- images
# ---- ...
# -- lishu
# -- zhuanshu
# -- kaishu