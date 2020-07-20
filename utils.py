import os
import re
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix


class Evaluator(object):
    '''
    精度评估
    '''

    def __init__(self, num_class):
        self.num_class = num_class
        self.conf_mat = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        '''
        分类正确的像素点数和所有的像素点数的比例
        '''
        Acc = np.diag(self.conf_mat).sum() / \
            self.conf_mat.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        '''
        计算每一类分类正确的像素点数和该类的所有像素点数的比例然后求平均
        '''
        Acc = np.diag(self.conf_mat) / \
            self.conf_mat.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        '''
        计算每一类的IoU然后求平均
        '''
        MIoU = np.diag(self.conf_mat) / (
            np.sum(self.conf_mat, axis=1) + np.sum(self.conf_mat, axis=0) -
            np.diag(self.conf_mat))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        '''
        根据每一类出现的频率对各个类的IoU进行加权求和
        '''
        freq = np.sum(self.conf_mat, axis=1) / \
            np.sum(self.conf_mat)
        iu = np.diag(self.conf_mat) / (
            np.sum(self.conf_mat, axis=1) + np.sum(self.conf_mat, axis=0) -
            np.diag(self.conf_mat))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def generate_matrix(self, gt_image, pre_image):
        conf_mat = confusion_matrix(
            gt_image.flatten(), pre_image.flatten(), np.arange(self.num_class))
        return conf_mat

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.conf_mat += self.generate_matrix(gt_image, pre_image)

    def reset(self):
        self.conf_mat = np.zeros((self.num_class,) * 2)


def blue(x): return '\033[94m' + x + '\033[0m'


def show_confMat(confusion_mat, classes_name, set_name, out_dir):
    """
    可视化混淆矩阵，保存png格式
    """
    # 归一化
    np.set_printoptions(precision=2)
    # confusion_mat[confusion_mat==0]=np.nan
    np.seterr(divide='ignore', invalid='ignore')
    confusion_mat_N = confusion_mat.astype(
        'float')/confusion_mat.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 8), dpi=120)
    # 打印数字
    ind_array = np.arange(len(classes_name))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = confusion_mat_N[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red',
                     fontsize=14, va='center', ha='center')
    # 获取颜色
    # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    cmap = plt.cm.get_cmap('Reds')
    plt.imshow(confusion_mat_N, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    # 设置文字
    xlocations = np.array(range(len(classes_name)))
    plt.xticks(xlocations, classes_name, rotation=90)
    plt.yticks(xlocations, classes_name)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix ' + set_name)
    # offset the tick
    tick_marks = np.array(range(len(classes_name)))+0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    # 保存
    plt.savefig(os.path.join(out_dir, '混淆矩阵' + set_name + '.png'))
    plt.close()


def smooth(csv_path, weight=0.6):
    data = pd.read_csv(filepath_or_buffer=csv_path, header=0, names=[
                       'Step', 'Value'], dtype={'Step': np.int, 'Value': np.float})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    save = pd.DataFrame({'Step': data['Step'].values, 'Value': smoothed})
    save.to_csv('smooth_'+re.split('[/.]', csv_path)[1]+'.csv')


def Normalization(array):
    '''
    输入数组减去均值，进行中心化处理
    min-max标准化，进行归一化处理
    '''
    array = np.transpose(array, (2, 0, 1))  # 通道数提前
    for i in range(array.shape[0]):
        mean = np.mean(array[i, :, :])
        array[i] = array[i]-mean
    array = np.transpose(array, (1, 2, 0))
    array = array.astype(float)
    array = (array-np.min(array))/(np.max(array)-np.min(array))
    return array


def pad(X, margin):
    '''
    填充边
    '''
    newX = np.zeros((X.shape[0]+margin*2, X.shape[1]+margin*2, X.shape[2]))
    newX[margin:X.shape[0]+margin, margin:X.shape[1]+margin, :] = X
    return newX


def randomCrop(img, mask, out_size):
    '''
    随机切片
    '''
    h, w = img.shape[0], img.shape[1]
    i = np.random.randint(0, h-out_size)
    j = np.random.randint(0, w-out_size)
    img_patch = img[i:i+out_size, j:j+out_size, :]
    mask_patch = mask[i:i+out_size, j:j+out_size]
    return img_patch, mask_patch


def getPredict(net, img, device):
    '''
    获取预测值图像
    '''
    net.eval()
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img)
        probs = F.softmax(output, dim=1)
        probs = probs.squeeze(0)
        probs = probs.cpu().numpy()
        probs = np.argmax(probs, 0)
    return probs
