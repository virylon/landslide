import logging
import os
import re
import scipy.io
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from utils import show_confMat, smooth, Evaluator, randomCrop, getPredict, Normalization, pad
from unet import UNet
from ulenet import ULeNet


dir_img = 'img/'
dir_mask = 'mask/'
dir_patch = 'patch/'
dir_predict = 'predict/'
dir_confmat = 'confmat/'
dir_scalar = 'scalar/'
dir_checkpoints='checkpoints/'
model = 'checkpoints/CP_epoch100.pth'

def createDir():
    if not os.path.exists(dir_img):
        os.makedirs(dir_img)
    if not os.path.exists(dir_mask):
        os.makedirs(dir_mask)
    if not os.path.exists(dir_patch):
        os.makedirs(dir_patch)
    if not os.path.exists(dir_predict):
        os.makedirs(dir_predict)
    if not os.path.exists(dir_confmat):
        os.makedirs(dir_confmat)
    if not os.path.exists(dir_scalar):
        os.makedirs(dir_scalar)
    if not os.path.exists(dir_checkpoints):
        os.makedirs(dir_checkpoints)

def createPatch():
    imgs = os.listdir(dir_img)
    patch_size = 128
    for img in imgs:
        image = cv2.imread(os.path.join(dir_img, img))#H W C
        # mask = cv2.imread(os.path.join(dir_mask, img))
        # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask=Image.open(os.path.join(dir_mask,img))
        mask=mask.convert('L')
        mask=np.array(mask)
        mask_class = np.unique(mask)
        for i in range(len(mask_class)):
            mask[mask == mask_class[i]] = i
        # 顺序分割
        for h in range(0, image.shape[0]-patch_size, patch_size):
            for w in range(0, image.shape[1]-patch_size, patch_size):
                img_patch = image[h:h+patch_size, w:w+patch_size, :]
                mask_patch = mask[h:h+patch_size, w:w+patch_size]
                patch_dict = {}
                patch_dict['image'] = img_patch
                patch_dict['mask'] = mask_patch
                dict_name = str(os.path.splitext(img)[0])+'_%d_%d.mat' % (h, w)
                scipy.io.savemat(os.path.join(
                    dir_patch, dict_name), patch_dict)
        # 随机分割
        # patch_num = (image.shape[0]*image.shape[1])//(patch_size*patch_size)
        # for i in range(patch_num*5):
        #     img_patch, mask_patch = randomCrop(image, mask, patch_size)
        #     patch_dict = {}
        #     patch_dict['image'] = img_patch
        #     patch_dict['mask'] = mask_patch
        #     dict_name = str(os.path.splitext(img)[0])+'_%d.mat' % i
        #     scipy.io.savemat(os.path.join(dir_patch, dict_name), patch_dict)


def showPatch():
    images = os.listdir(dir_patch)
    for img in images:
        mat = scipy.io.loadmat(os.path.join(
            dir_patch, os.path.splitext(img)[0] + '.mat'))
        image = mat['image']
        mask = mat['mask']
        # image = Normalization(image)
        _, ax = plt.subplots(1, 2)
        ax[0].set_title('image')
        ax[0].imshow(image)
        ax[1].set_title('mask')
        ax[1].imshow(mask)
        plt.show()


def soomthScalar():
    scalars = os.listdir(dir_scalar)
    for scalar in scalars:
        smooth(os.path.join(dir_scalar, scalar))


def predict():
    net = UNet(n_channels=3, n_classes=2)
    # net = ULeNet(n_channels=3, n_classes=6)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))
    evaluator = Evaluator(num_class=2)
    evaluator.reset()
    for img in os.listdir(dir_img):
        image = cv2.imread(os.path.join(dir_img, img))
        # mask = cv2.imread(os.path.join(dir_mask, img))
        # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask=Image.open(os.path.join(dir_mask,img))
        mask=mask.convert('L')
        mask=np.array(mask)
        mask_class = np.unique(mask)
        for i in range(len(mask_class)):
            mask[mask == mask_class[i]] = i
        predict = np.zeros((image.shape[0], image.shape[1]))
        p_size = 128
        for i in range(0, image.shape[0]-p_size, p_size):
            for j in range(0, image.shape[1]-p_size, p_size):
                patch = image[i:i+p_size, j:j+p_size, :]
                patch = Normalization(patch)
                predict[i:i+p_size, j:j+p_size] = getPredict(
                    net=net, img=patch, device=device)
        evaluator.add_batch(mask, predict)
        # mIoU = evaluator.Mean_Intersection_over_Union()
        predict = Image.fromarray((predict).astype(np.uint8))
        predict.save(os.path.join(
            dir_predict, os.path.splitext(img)[0] + '.tif'))
    show_confMat(evaluator.conf_mat, [str(c)for c in range(
        2)], re.split('[/.]', model)[1], dir_confmat)


def showPredict():
    images = os.listdir(dir_predict)
    for img in images:
        pred = cv2.imread(os.path.join(dir_predict, img), cv2.IMREAD_GRAYSCALE)
        pred_class = np.unique(pred)
        for i in range(len(pred_class)):
            pred[pred == pred_class[i]] = i
        image = cv2.imread(os.path.join(dir_img, img))
        # mask = cv2.imread(os.path.join(dir_mask, img))
        # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask=Image.open(os.path.join(dir_mask,img))
        mask=mask.convert('L')
        mask=np.array(mask)
        mask_class = np.unique(mask)
        for i in range(len(mask_class)):
            mask[mask == mask_class[i]] = i
        _, ax = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(wspace=0.2, hspace=0)  # 调整子图间距
        ax[0].set_title('image')
        ax[0].imshow(image)
        ax[1].set_title('mask')
        ax[1].imshow(mask)
        ax[2].set_title('predict')
        ax[2].imshow(pred)
        savefig_path = os.path.join(
            dir_confmat, '预测图' + os.path.splitext(img)[0] + '.png')
        plt.savefig(savefig_path, dpi=300)


if __name__ == "__main__":
    # createDir()
    # createPatch()  # create patch{image,mask}
    # showPatch()  # show patch{image,mask}
    # soomthScalar()#smooth scalar
    # predict()
    showPredict()  # show image mask predict
