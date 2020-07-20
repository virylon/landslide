import logging
import os
import re
import scipy.io
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch


dir_img = 'img/'
dir_mask = 'mask/'

imgs = os.listdir(dir_img)
# for img in imgs:
#     image = cv2.imread(os.path.join(dir_img, img))
#     mask=Image.open(os.path.join(dir_mask,img))
#     mask=mask.convert('L')
#     mask=np.array(mask)
#     # mask = cv2.imread(os.path.join(dir_mask, img))
#     # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    
#     mask_class = np.unique(mask)
#     for i in range(len(mask_class)):
#         mask[mask == mask_class[i]] = i
# plt.imshow(mask)
# plt.show()
