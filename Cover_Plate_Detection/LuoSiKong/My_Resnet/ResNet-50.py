import numpy as np
import os
import cv2
import random
random.seed(5)
import matplotlib.pyplot as plt
import time

from Utility import ResNet50
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# 参数
batch_size = 24
epochs = 100
input_size = (320, 200)
# 图片总数 NG+OK
num_pic = 574+600
# 图片类型
num_classes = 2

# 提取数据
x_train = []
y_train = []
x_test = []
y_test = []
x_val = []
y_val = []
class_name = ['NG', 'OK']
num = 0
img_name_test = []
def read_directory_for_train(directory_name,num_folder):
    i = 0
    for filename in os.listdir(directory_name):
        if num_folder == 2:
            i += 1
            if i % 6 > 0:
                continue
        # img = cv2.imread(directory_name + "/" + filename, -1)
        img = cv2.imdecode(np.fromfile(directory_name + "/" + filename, dtype=np.uint8), -1)
        img = cv2.resize(img, (input_size[1], input_size[0]))
        # (x, y, z) = img.shape
        label = [0] * num_classes
        label[num_folder] = 1
        r = random.random()
        if r < 0.2:
            x_val.append(img/255)
            y_val.append(label)
        elif r < 0.4:
            x_test.append(img/255)
            y_test.append(label)
            img_name_test.append(filename)
        else:
            x_train.append(img/255)
            y_train.append(label)

def read_directory_for_test(directory_name, num_folder):
    i = 0
    for filename in os.listdir(directory_name):
        if num_folder == 2:
            i += 1
            if i % 6 > 0:
                continue
        # img = cv2.imread(directory_name + "/" + filename, -1)
        img = cv2.imdecode(np.fromfile(directory_name + "/" + filename, dtype=np.uint8), -1)
        img = cv2.resize(img, (input_size[1], input_size[0]))
        # (x, y, z) = img.shape
        label = [0] * num_classes
        label[num_folder] = 1
        x_test.append(img / 255)
        y_test.append(label)
        img_name_test.append(filename)
# 随机数据增强
# for l in range(2):
#     p = random.random()
#     if p < 1/4:
#         img1 = cv2.flip(img, 1)          # 左右翻转
#         x_train.append(img1)
#         y_train.append(label)
#     elif p < 2/4:
#         img2 = cv2.flip(img, 2)            # 上下翻转
#         x_train.append(img2)
#         y_train.append(label)
#     elif p < 3/5:
#         img3 = cv2.resize(img, (200, 200))  # 放大图片至200，取中间的128
#         img3 = img3[36:164, 36:164, :]
#         x_train.append(img3)
#         y_train.append(label)
#     elif p < 3/4:
#         img4 = img                           # 颜色通道变换
#         img4[:, :, 0] = img[:, :, 1]
#         img4[:, :, 1] = img[:, :, 2]
#         img4[:, :, 2] = img[:, :, 0]
#         x_train.append(img4)
#         y_train.append(label)
#     else:
#         img5 = tf.image.random_brightness(img, max_delta=0.1)  # 亮度随机
#         x_train.append(img5)
#         y_train.append(label)

# read_directory_for_test(r'C:/Users/ASUS/Desktop/Using AI/螺丝孔分类/data/4.16/NG/LeftDown', 0)
# read_directory_for_test(r'C:/Users/ASUS/Desktop/Using AI/螺丝孔分类/data/4.16/OK/LeftDown', 1)
# read_directory_for_test(r'C:/Users/ASUS/Desktop/Using AI/螺丝孔分类/data/4.16/NG/LeftUp', 0)
# read_directory_for_test(r'C:/Users/ASUS/Desktop/Using AI/螺丝孔分类/data/4.16/OK/LeftUp', 1)
# read_directory_for_test(r'C:/Users/ASUS/Desktop/Using AI/螺丝孔分类/data/4.16/NG/Middle', 0)
# read_directory_for_test(r'C:/Users/ASUS/Desktop/Using AI/螺丝孔分类/data/4.16/OK/Middle', 1)
read_directory_for_test(r'C:/Users/ASUS/Desktop/Using AI/螺丝孔分类/data/4.16/NG/Right', 0)
read_directory_for_test(r'C:/Users/ASUS/Desktop/Using AI/螺丝孔分类/data/4.16/OK/Right', 1)


x_test = np.array(x_test)
x_test = np.expand_dims(x_test, axis=-1)
y_test = np.array(y_test)
# x_train = np.array(x_train)
# x_train = np.expand_dims(x_train, axis=-1)
# y_train = np.array(y_train)
# x_val = np.array(x_val)
# x_val = np.expand_dims(x_val, axis=-1)
# y_val = np.array(y_val)
# np.random.seed(120)  # 设置随机种子，让每次结果都一样，方便对照
# np.random.shuffle(x_train)  # 使用shuffle()方法，让输入x_train乱序
# np.random.seed(120)
# np.random.shuffle(y_train)
# print(sum(y_train))
# print(sum(y_val))
print(sum(y_test))


# 建立网络
model = ResNet50(input_shape=(input_size[0], input_size[1], 1), classes=num_classes)
# model.compile(optimizer=keras.optimizers.Adam(lr=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])


# 读取模型
# 参数默认 by_name = Fasle， 否则只读取匹配的权重
model.load_weights(r'.h5/ResNet_1.3.h5', by_name='True')


# 训练
# early_stopping = EarlyStopping(monitor='val_loss', patience=3)
# Reduce=ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=1,verbose=1,mode='auto',epsilon=0.001,cooldown=0,min_lr=0)
# history = model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_val,y_val),callbacks=[early_stopping,Reduce])
# callbacks=[ ]
# print(history.history)

# 保存模型
# model.save('.h5/ResNet_1.4.h5')

# recall,precision,auc,F1
# from sklearn.metrics import roc_auc_score
# x_pre = x_test.reshape((-1,input_size[0],input_size[1],1))
# yy_pred = model.predict(x_pre)
# y_pred = np.argmax(yy_pred, axis=1)
# y_t = np.argmax(y_test, axis=1)
# TP = np.sum(np.logical_and(np.equal(y_t, 1), np.equal(y_pred, 1)))
# FP = np.sum(np.logical_and(np.equal(y_t, 0), np.equal(y_pred, 1)))
# TN = np.sum(np.logical_and(np.equal(y_t, 0), np.equal(y_pred, 0)))
# FN = np.sum(np.logical_and(np.equal(y_t, 1), np.equal(y_pred, 0)))
# precision = TP / (TP + FP)
# recall = TP / (TP + FN)
# accuracy = (TP + TN) / (TP + FP + TN + FN)
# F1_Score = 2 * precision * recall / (precision + recall)
# auc = roc_auc_score(y_t,y_pred)
# print('accuracy:'+str(accuracy)+' recall:'+str(recall)+' precision:'+str(precision)+' auc:'+str(auc)+' f1:'+str(F1_Score))

# 测试结果
correct = 0
wrong = [0] * num_classes
for i in range(len(x_test)):
    x_pre = x_test[i].reshape((1,input_size[0],input_size[1],1))
    # time_start = time.time()  # 开始计时
    y_pre = model.predict(x_pre)
    # time_end = time.time()  # 结束计时
    # time_c = time_end - time_start  # 运行所花时间
    # print('time cost', time_c, 's')
    if y_pre[0][0] > 0.3:
        classes = 0
    else:
        classes = 1
    # print(i)
    # print(y_pre)
    # if y_test[i][classes] == 1:
    #     correct = correct + 1
    # else:
    wrong[classes] = wrong[classes] + 1
    # print(class_name[np.argmax(y_test[i])], class_name[np.argmax(y_pre)])
    # cv2.imwrite('C:/Users/ASUS/Desktop/Using_AI/LuoSiKong/test_result/4.16/LeftDown/' + str(class_name[classes]) + '_' + img_name_test[i] + '.jpg', x_test[i]*255)
    # cv2.imwrite('C:/Users/ASUS/Desktop/Using_AI/LuoSiKong/test_result/4.16/LeftUp/' + str(class_name[classes]) + '_' + img_name_test[i] + '.jpg', x_test[i]*255)
    # cv2.imwrite('C:/Users/ASUS/Desktop/Using_AI/LuoSiKong/test_result/4.16/Middle/' + str(class_name[classes]) + '_' + img_name_test[i] + '.jpg', x_test[i]*255)
    cv2.imwrite('C:/Users/ASUS/Desktop/Using_AI/LuoSiKong/test_result/4.16/Right/' + str(class_name[classes]) + '_' + img_name_test[i] + '.jpg', x_test[i]*255)

    # cv2.imwrite('C:/Users/ASUS/Desktop/Using_AI/LuoSiKong/fault_img/4.16/Middle_' + str(class_name[np.argmax(y_test[i])])
    #             + '_' + str(class_name[classes]) + '_' + img_name_test[i] + '.jpg', x_test[i]*255)
    # cv2.imwrite('C:/Users/ASUS/Desktop/Using_AI/LuoSiKong/fault_img/4.16/Right_' + str(class_name[np.argmax(y_test[i])])
    #               + '_' + str(class_name[classes]) + '_' + img_name_test[i] + '.jpg', x_test[i] * 255)
    # cv2.imwrite('C:/Users/ASUS/Desktop/Using_AI/LuoSiKong/test_result/4.16/Mix_' + str(class_name[np.argmax(y_test[i])])
    #               + '_' + str(class_name[classes]) + '_' + img_name_test[i] + '.jpg', x_test[i] * 255)
    # plt.imshow(x_test[i],cmap='gray')
    # plt.show()
print('test Result:',correct,'/', len(y_test), correct/len(y_test))
print('fault distribution',wrong)

# 画图
# val_acc = history.history['val_acc']
# val_loss = history.history['val_loss']
# acc = history.history['acc']
# loss = history.history['loss']
# epoch = range(1,len(acc)+1)
# #第一行第一列图形
# ax1 = plt.subplot(2,2,1)
# #第一行第二列图形
# ax2 = plt.subplot(2,2,2)
# ax3 = plt.subplot(2,2,3)
# ax4 = plt.subplot(2,2,4)
# #选择ax1
# plt.sca(ax1)
# #绘制红色曲线
# plt.plot(epoch,loss,'b',label='train loss')
# plt.legend()
# plt.sca(ax2)
# plt.plot(epoch,acc,'b',label='train acc')
# plt.legend()
# plt.sca(ax3)
# plt.plot(epoch,val_loss,'b',label='val loss')
# plt.legend()
# plt.sca(ax4)
# plt.plot(epoch,val_acc,'b',label='val acc')
# plt.legend()
# plt.show()


# Voting
# correct = 0
# wrong = [0] * num_classes
# for i in range(len(x_test)):
#     img = x_test[i]
#     imgs = np.zeros((3,256,256,1))
#     imgs[2] = img
#     imgs[0] = cv2.flip(img, 1).reshape((256,256,1))        # 数据增强：左右翻转
#     imgs[1] = cv2.flip(img, 2).reshape((256,256,1))        # 数据增强: 上下翻转
#     y = model.predict(imgs)
#     sum_pred = sum(y)
#     classes = np.argmax(sum_pred)
#     classes = np.argmax(y)
#     if y_test[i][classes] == 1:
#         correct = correct + 1
#     else:
#         wrong[classes] = wrong[classes] + 1
# print('Voting Result:',correct, len(y_test))
# print(wrong)

