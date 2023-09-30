import os
import numpy as np
import cv2

# # load dataset
# phase3_dir = 'D:/高源/BU/AICV Lab/BML Generate/Phase3'
# imgs = []
# labels = []
# for patient_name in os.listdir(phase3_dir):
#     if patient_name == '9215390':
#         break
#     for img_name in os.listdir(phase3_dir + '/' + patient_name + '/' + 'v00/meta/images'):
#         img_add = phase3_dir + '/' + patient_name + '/' + 'v00/meta/images/' + img_name
#         img = cv2.imdecode(np.fromfile(img_add, dtype=np.uint8), 0)
#         img = cv2.resize(img, (280, 280), interpolation=cv2.INTER_AREA)
#         imgs.append(img)
#         label = 0
#         if 'bml_masks' in os.listdir(phase3_dir + '/' + patient_name + '/' + 'v00/meta/'):
#             if img_name[:-4]+'_mask'+'.bmp' in os.listdir(phase3_dir + '/' + patient_name + '/' + 'v00/meta/bml_masks'):
#                 label = 1
#         labels.append(label)
# X_train = np.array(imgs)
# Y_train = np.array(labels)

import matplotlib.pyplot as plt
import numpy as np
import time
from math import *

plt.ion() #开启interactive mode 成功的关键函数
plt.figure(1)
t = [0]
t_now = 0
m = [sin(t_now)]

for i in range(5):
    plt.clf() #清空画布上的所有内容
    t.append(i)#模拟数据增量流入，保存历史数据
    m.append(2*i)#模拟数据增量流入，保存历史数据
    plt.plot(t,m,'-r')
    # plt.draw()#注意此函数需要调用
    # time.sleep(1)
    plt.pause(1)