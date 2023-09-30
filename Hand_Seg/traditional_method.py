import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import data,filters,img_as_ubyte

image = cv2.imdecode(np.fromfile('D:/BU_CV/Hand_OA/remove_BG/9052207.png', dtype=np.uint8), -1)
sample = cv2.imdecode(np.fromfile('C:/代码/BU/Carmine/data/train/0/9000798_pip2.png', dtype=np.uint8), -1)
image = cv2.resize(image, (1000, 1600))


# sobel边缘检测
sobel_image = cv2.Sobel(image, cv2.CV_16S, 0, 1)
sobel_image = cv2.convertScaleAbs(sobel_image)
# sobel_image[sobel_image > 50] = 255
sobel_sample = cv2.Sobel(sample, cv2.CV_16S, 0, 1)
sobel_sample = cv2.convertScaleAbs(sobel_sample)
sobel_sample = sobel_sample[15:165, 30:]
sobel_sample = cv2.resize(sobel_sample, (80, 80))
sobel_sample = cv2.blur(sobel_sample,(5,5))


# 去噪 threshold + erode
# edges = filters.sobel(sample)
# edges = img_as_ubyte(edges)
# # Threshold for sample
# _,Thresh = cv2.threshold(edges,11,120,cv2.THRESH_BINARY)
# # Open operation for sample
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
# sample = cv2.erode(Thresh, kernel)
# sample = cv2.dilate(eroded, kernel)

coef = cv2.matchTemplate(sobel_sample, sobel_image, cv2.TM_CCOEFF_NORMED)
print(coef.shape)

# for i in range(3):
#     max = np.unravel_index(coef[:900, 200:].argmax(), coef[:900, 200:].shape)
#     coef[max[0]-50:max[0]+50, max[1]+200-70:max[1]+200+70] = 0
#     image[max[0] + 75 - 5:max[0] + 75 + 5, max[1] + 275 - 5:max[1] + 275 + 5] = 0



# based on region
dip2_coef = np.unravel_index(coef[150:300, 150:300].argmax(), coef[150:300, 150:300].shape)
pip2_coef = np.unravel_index(coef[300:500, 150:300].argmax(), coef[300:500, 150:300].shape)
mcp2_coef = np.unravel_index(coef[600:750, 150:300].argmax(), coef[600:750, 150:300].shape)
image[dip2_coef[0]+75+150-5:dip2_coef[0]+75+150+5, dip2_coef[1]+75+150-5:dip2_coef[1]+75+150+5] = 0
image[pip2_coef[0]+75+300-5:pip2_coef[0]+75+300+5, pip2_coef[1]+75+150-5:pip2_coef[1]+75+150+5] = 0
image[mcp2_coef[0]+75+600-5:mcp2_coef[0]+75+600+5, mcp2_coef[1]+75+150-5:mcp2_coef[1]+75+150+5] = 0

ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
plt.sca(ax1)
plt.title('coef')
plt.imshow(coef, 'gray')
plt.sca(ax2)
plt.title('image')
plt.imshow(image, 'gray')
plt.show()

















