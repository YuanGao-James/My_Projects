import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import cv2

dir_path = 'D:/Using AI/烟弹管缺陷检测/Data/6.10下午4点'

mini = []
maxi = []
aver = []
time = []
l = os.listdir(dir_path)
l.sort(key=lambda x: int(x.split('-')[0]))
for filename in l:
    if filename[-4:] == 'jpg':
        continue
    if filename == '92-2022-06-10-16-05-41.830':
        break
    print(filename)
    txt_path = os.path.join(dir_path, filename)
    l = []
    with open(txt_path, 'r') as f:
        annotations = f.readlines()
        i = 0
        for ann in annotations:
            ann = list(map(float, ann.split()))
            l = l + ann
    img = np.array(l,dtype='uint8')
    img = img.reshape(512, 640)

    mini.append(np.min(img))
    maxi.append(np.max(img))
    aver.append(round(np.mean(img), 2))
    time.append(filename[13:-8])

    sns.heatmap(img, cmap="rainbow")
    plt.text(500,-20,'min:'+str(np.min(img))+' max:'+str(np.max(img))+' mean:'+str(round(np.mean(img), 2)),
             fontsize=10,verticalalignment="top",horizontalalignment="right")
    plt.axis('off')
    myfig = plt.gcf()
    myfig.savefig(dir_path+'/'+ str(filename[13:-4])+'.jpg', dpi=300)
    plt.clf()

plt.plot(time, mini, color='r', label='min')
plt.plot(time, maxi, color='g', label='max')
plt.plot(time, aver, color='b', label='mean')
plt.legend()
plt.xticks(rotation=90)
import matplotlib.ticker as ticker
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))
# plt.show()
myfig = plt.gcf()
myfig.savefig(dir_path+'/'+dir_path.split('/')[-1]+'.jpg', dpi=300)
plt.clf()



# plt.imshow(img_list[1]-img_list[0])
# print(img_list[1]-img_list[0])
# myfig = plt.gcf()
# myfig.savefig('D:/Using AI/烟弹管缺陷检测/Data/温度_6.7/1.jpg', dpi=300)


# plt.figure(figsize=(20, 20))
# for i in range(1, 10):
#     for j in range(1, 10):
#         # if i == 1 and j == 1:
#         #     ax = plt.subplot(10, 10, 1)
#         #     plt.title(time[0])
#         #     plt.axis('off')
#         #     plt.text(-10,200,time[0])
#         if i >= j:
#             continue
#         ax = plt.subplot(10, 10, (j-1)*10+i)
#         plt.sca(ax)
#         plt.axis('off')
#         if j == 1:
#             plt.title(time[i-1])
#         plt.xlabel(time[i-1])
#         plt.imshow(img_list[i-1]-img_list[j-1])
#
# plt.show()

