from PIL import Image
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from xml2txt import convert_annotation

def matching(sample,image):
    try:
        image2 = cv2.equalizeHist(image)

        # 对image进行模板匹配
        img_list = []
        COEF = []
        points = []
        for i in range(20, 28):
            for j in range(5):
                small_sample = image2[i * 20:i * 20 + 228, 300 + j * 20:1165 + j * 20]
                img_list.append(image2[i * 20 - 100:i * 20 + 328, 100 + j * 20:1165 + j * 20])
                coef = cv2.matchTemplate(small_sample, sample, cv2.TM_CCORR_NORMED)
                COEF.append(coef[0])
                points.append((100 + j * 20, i * 20 - 100))
        idx = np.argmax(COEF)
        return points[idx]
    except:
        print('matching failed')
        return 0

def matching_output_result(input_dir,output_dir):
    get_sample = cv2.imdecode(
        np.fromfile('D:/Using AI/烟弹管缺陷检测/Data/4.26/A3138MG000_7K01D72PAK00001/1/Pic_2022_04_26_184436_142.bmp', dtype=np.uint8), -1)
    get_sample = cv2.equalizeHist(get_sample)
    sample = get_sample[435:663, 360:1225]
    get_sample = cv2.imdecode(
        np.fromfile('D:/Using AI/烟弹管缺陷检测/Data/4.26/A3138MG000_7K01D72PAK00001/1/Pic_2022_04_26_184446_169.bmp', dtype=np.uint8), -1)
    get_sample = cv2.equalizeHist(get_sample)
    sample_empty = get_sample[435:663, 360:1225]

    for filename in os.listdir(input_dir):
        if filename[-3:] != 'bmp':
            continue
        try:
            image = cv2.imdecode(np.fromfile(input_dir + '/' + filename, dtype=np.uint8), -1)
            image2 = cv2.equalizeHist(image)

            # 对image进行模板匹配
            ballanced_img_list = []
            oringin_img_list = []
            COEF = []
            points = []
            for i in range(20, 28):
                for j in range(5):
                    small_sample = image2[i * 20:i * 20 + 228, 300 + j * 20:1165 + j * 20]
                    oringin_img_list.append(image[i * 20 - 100:i * 20 + 328, 100 + j * 20:1165 + j * 20])
                    ballanced_img_list.append(image2[i * 20 - 100:i * 20 + 328, 100 + j * 20:1165 + j * 20])
                    coef = cv2.matchTemplate(small_sample, sample, cv2.TM_CCORR_NORMED)
                    COEF.append(coef[0])
                    points.append((100 + j * 20, i * 20))
            idx = np.argmax(COEF)
            cut_img = oringin_img_list[idx]
            # ballanced_cut_img = ballanced_img_list[idx]
            cv2.imencode('.jpg', cut_img)[1].tofile(output_dir + '/' + filename)

            # coef2 = cv2.matchTemplate(ballanced_cut_img[100:-100, 200:], sample_empty, cv2.TM_CCORR_NORMED)
            # if max(COEF) > coef2:
            #     print(input_dir+'/'+filename, max(COEF),coef2 )
            #     cv2.imencode('.jpg', cut_img)[1].tofile(output_dir + '/' + filename)
            # else:
            #     print(input_dir+'/'+filename, max(COEF),coef2, 'reject')
        except:
            print('failed:'+input_dir+'/'+filename)
    # return points[idx]

def convert(size, box):
    # size=(width, height)  b=(xmin, xmax, ymin, ymax)
    # x_center = (xmax+xmin)/2        y_center = (ymax+ymin)/2
    # x = x_center / width            y = y_center / height
    # w = (xmax-xmin) / width         h = (ymax-ymin) / height

    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    x = x_center / size[0]
    y = y_center / size[1]

    w = (box[1] - box[0]) / size[0]
    h = (box[3] - box[2]) / size[1]

    # print(x, y, w, h)
    return (x, y, w, h)
def convert_annotation(xml_files_path, save_txt_files_path, img_path):
    xml_files = os.listdir(xml_files_path)
    get_sample = cv2.imdecode(
        np.fromfile('D:/Using AI/烟弹管缺陷检测/Data/4.26/A3138MG000_7K01D72PAK00001/1/Pic_2022_04_26_184436_142.bmp', dtype=np.uint8), -1)
    get_sample = cv2.equalizeHist(get_sample)
    sample = get_sample[435:663, 360:1225]
    for xml_name in xml_files:
        if xml_name[-3:] == 'bmp':
            continue
        # elif xml_name[-3:] == 'xml':
        #     continue
        image = cv2.imdecode(np.fromfile(img_path + '/' + xml_name[:-4] +'.bmp', dtype=np.uint8), -1)
        points = matching(sample, image)
        # print(points)
        xml_file = os.path.join(xml_files_path, xml_name)
        out_txt_path = os.path.join(save_txt_files_path, xml_name.split('.')[0] + '.txt')
        out_txt_f = open(out_txt_path, 'w')
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        # w = int(size.find('width').text)
        # h = int(size.find('height').text)
        w = 1065
        h = 428

        for obj in root.iter('object'):
            cls_id = obj.find('name').text
            if cls_id == '0' or cls_id == '19':
                continue
            if cls_id == '2':
                cls_id = '1'
            # if cls_id == '14':
            #     print(xml_name)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text)-points[0], float(xmlbox.find('xmax').text)-points[0],
                 float(xmlbox.find('ymin').text)-points[1], float(xmlbox.find('ymax').text)-points[1])
            # b=(xmin, xmax, ymin, ymax)
            bb = convert((w, h), b)
            # print(bb)
            out_txt_f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
def xml_convert(xml_path, txt_path, img_path):
    os.makedirs(txt_path) if os.path.exists(txt_path) == False else 0
    # 1、需要转化的类别
    classes = ['bian', 'heidian', 'liaohua', 'guankouquejiao', 'duanzhen', 'wuneiguan', 'shuikougao', 'neiguanquejiao',
               'baiseqipao', 'shuiyin', 'tuoshang', 'jiaxian', 'yueyayashang', 'youlieheng', 'guankoupifeng', 'toubuchuankong',
               'toubupifeng', 'toubusuoshui', 'others', 'background'] #注意：这里根据自己的类别名称及种类自行更改
    convert_annotation(xml_path, txt_path, img_path)


if __name__ == '__main__':
    print(i for i in range(3))


# plt.plot(range(60),COEF,'b',label='train acc')
# plt.show()

# ax1 = plt.subplot(2,2,1)
# ax2 = plt.subplot(2,2,2)
# plt.sca(ax1)
# plt.imshow(img_list[idx],'gray')
# plt.sca(ax2)
# plt.imshow(sample,'gray')

# plt.show()