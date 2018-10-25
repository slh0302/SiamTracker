# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 10/8/18 2:59 PM

import cv2
import os
import numpy as np
from mcode.MouseTrack.utils.detector  import faster_rcnn

gpus = '14'
os.environ['CUDA_VISIBLE_DEVICES'] = gpus
fst = faster_rcnn(1, False, base_net=101, load_name='/home/slh/tf-project/mouse/faster_rcnn/model/res101/pascal_voc/faster_rcnn_1_20_233.pth')
img = np.array(cv2.imread('/home/slh/test/000271.jpg'))
res = fst.detect_img(img, gpus_number=gpus)
print(res)


# do tracking

