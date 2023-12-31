import cv2
import os
from glob import glob
import random
import matplotlib.pyplot as plt 
# import argparse
from tqdm import tqdm
import numpy as np
#用法说明 https://zhuanlan.zhihu.com/p/350335747
# parser = argparse.ArgumentParser()
# parser.add_argument('--root',type=str ,default='D:/Using AI/烟弹管缺陷检测/Data/4.26', help="which should include ./pic and ./labels and label.txt")
# parser.add_argument('--dt',type=str ,default='D:/Using AI/烟弹管缺陷检测/Data/4.26', help="yolo format results of detection, include ./labels")
# parser.add_argument('--conf' , type=float ,default=0.5, help="visulization conf thres")
# arg = parser.parse_args()

colorlist = []
# 5^3种颜色。
for i in range(25,256,50):
    for j in range(25,256,50):
        for k in range(25,256,50):
            colorlist.append((i,j,k))
random.shuffle(colorlist)

def plot_bbox(img_path, img_dir, out_dir, gt=None ,dt=None, cls2label=None, line_thickness=None):
    conf = 0.5
    if img_path[-4:] != '.png':
        return
    img = cv2.imdecode(np.fromfile(os.path.join(img_dir, img_path), dtype=np.uint8), 1)
    height, width,_ = img.shape
    tl = line_thickness or round(0.002 * (width + height) / 2) + 1  # line/font thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    if gt:
        tf = max(tl - 1, 1)  # font thickness
        with open(gt,'r') as f:
            annotations = f.readlines()
            # print(annotations,1)
            for ann in annotations:
                ann = list(map(float,ann.split()))
                ann[0] = int(ann[0])
                # print(ann,2)
                cls,x,y,w,h = ann
                color = colorlist[cls]
                c1, c2 = (int((x-w/2)*width),int((y-h/2)*height)), (int((x+w/2)*width), int((y+h/2)*height))
                cv2.rectangle(img, c1, c2, color, thickness=tl*2, lineType=cv2.LINE_AA)
                # 类别名称显示
                cv2.putText(img, str(cls2label[cls]), (c1[0], c1[1] - 2), 0, tl / 4, color, thickness=tf, lineType=cv2.LINE_AA)
    if dt:
        with open(dt,'r') as f:
            annotations = f.readlines()
            # print(annotations)
            for ann in annotations:
                ann = list(map(float,ann.split()))
                ann[0] = int(ann[0])
                # print(ann)
                if len(ann) == 6:
                    cls,x,y,w,h,conf = ann
                    if conf < arg.conf:
                        # thres = 0.5
                        continue
                elif len(ann) == 5:
                    cls,x,y,w,h = ann
                color = colorlist[len(colorlist) - cls - 1]

                c1, c2 = (int((x-w/2)*width), int((y-h/2)*height)), (int((x+w/2)*width), int((y+h/2)*height))
                cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

                # # cls label
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(cls2label[cls], 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                if len(ann) == 6:
                    cv2.putText(img, str(round(conf,2)), (c1[0], c1[1] - 2), 0, tl / 4, color, thickness=tf, lineType=cv2.LINE_AA)
    cv2.imencode('.png', img)[1].tofile(os.path.join(out_dir,img_path))
    # cv2.imwrite(os.path.join(out_dir,img_path),img)
def plot_box_on_pic(root):
    root_path = root
    pred_path = root
    img_dir = os.path.join(root_path, 'pic')
    GT_dir = os.path.join(root_path, 'labels')
    DT_dir = os.path.join(pred_path)
    out_dir = os.path.join(root_path, 'outputs')
    cls_dir = os.path.join(root_path,'label.txt')
    cls_dict = {}

    if not os.path.exists(img_dir):
        raise Exception("image dir {} do not exist!".format(img_dir))
    if not os.path.exists(cls_dir):
        raise Exception("class dir {} do not exist!".format(cls_dir))
    else:
        with open(cls_dir, 'r') as f:
            classes = f.readlines()
            for i in range(len(classes)):
                cls_dict[i] = classes[i].strip()
            print("class map:", cls_dict)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(GT_dir):
        print(f"WARNNING: {GT_dir} ,GT NOT Available!")
    if not os.path.exists(DT_dir):
        print(f"WARNNING: {DT_dir} ,DT NOT Available!")
    for each_img in tqdm(os.listdir(img_dir)):
        gt = None
        dt = None
        if os.path.exists(os.path.join(GT_dir, each_img.replace('png', 'txt'))):
            gt = os.path.join(GT_dir, each_img.replace('png', 'txt'))
        if os.path.exists(os.path.join(DT_dir, each_img.replace('png', 'txt'))):
            dt = os.path.join(DT_dir, each_img.replace('png', 'txt'))
        plot_bbox(each_img, img_dir, out_dir, gt, dt, cls2label=cls_dict)


if __name__ == "__main__":
    plot_box_on_pic('C:/代码/BU/hand_seg/YuanGao/yolov5-master/My data/visualization')
