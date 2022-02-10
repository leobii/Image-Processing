import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# 讀取(灰階)圖片
img_org = cv2.imread("./Dataset_OpenCvDl_Hw1/Q3_Image/House.jpg")
img = img_org[:, :, [2, 1, 0]]

def gaussian_blur_2():
    grayscale_average_img = np.mean(img, axis=2)
    # (axis=0 would average across pixel rows and axis=1 would average across pixel columns.)
    img_gaussian_filter = Gaussian_Filter(grayscale_average_img)

    def Gaussian_Filter(image):
        KernalSize = 3
        # 3*3 Gassian filter
        x, y = np.mgrid[-1:KernalSize-1, -1:KernalSize-1]
        gaussian_kernel = np.exp(-(x**2+y**2))
        #Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        """
        Convolve 3*3 Gassian Kernel with Image
        """
        resx = cv2.filter2D(image, -1, kernel=gaussian_kernel)
        #resx = signal.convolve2d(image, gaussian_kernel, boundary='symm', mode='same') #卷積
        return resx

    """
    Plot images
    """
    plt.figure(num='Gaussian Filter', figsize=(12, 9))  # 創建一個名為Gaussian Filter的窗口,並設置其大小

    plt.subplot(1, 3, 1)  # 將窗口分為兩行兩列共四的子圖，則可顯示四幅圖片。此為第一行第一列的圖
    plt.imshow(img)
    plt.title('House.jpg')
    plt.axis('off')

    plt.subplot(1, 3, 2)  # 第二個子圖(第二行第一列的圖)
    plt.imshow(grayscale_average_img, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')
    plt.show()

    plt.subplot(1, 3, 3)  # 第三個子圖(第一行第二列的圖)
    plt.imshow(img_gaussian_filter, cmap='gray')
    plt.title('Gaussian Blur')
    plt.axis('off')

    plt.show()  # 顯示窗口


def sobel_x():
    sobel_x, _ = sobel('dx')
    plt.imshow(sobel_x, cmap='gray')
    plt.title('Sobel X')
    plt.axis('off')
    plt.show()

def sobel_y():
    sobel_y, _ = sobel('dy')
    plt.imshow(sobel_y, cmap='gray')
    plt.title('Sobel Y')
    plt.axis('off')
    plt.show()

def magnitude():
    sobel_m, _ = sobel('magnitude')
    plt.imshow(sobel_m, cmap='gray')
    plt.title('Magnitude')
    plt.axis('off')
    plt.show()

def sobel(filtering_type):
    h, w, d = img.shape
    horizontal = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]  #----- to do ---------------#
    vertical = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]  #----- to do ---------------#
    Gx = np.zeros((h - 2, w - 2))
    Gy = np.zeros((h - 2, w - 2))

    for img_region, i, j in iterate_regions(img, 3):
        if filtering_type == 'dx':
            Gx[i, j] += np.sum(img_region * horizontal)  # ----- to do -------------- #
        elif filtering_type == 'dy':
            Gy[i, j] += np.sum(img_region * vertical)  # ----- to do -------------- #
        elif filtering_type == 'magnitude':
            Gx[i, j] += np.sum(img_region * horizontal)  # ----- to do -------------- #
            Gy[i, j] += np.sum(img_region * vertical)  # ----- to do -------------- #

    gradient = np.sqrt(np.square(Gx) + np.square(Gy))  # ----- to do --------------
    gradient = np.pad(gradient, (1, 1), 'constant')
    angle = np.arctan(Gy / Gx)  # ----- to do -------------- #
    angle = np.pad(angle, (1, 1), 'constant')

    output = np.clip(gradient,0,255)
    angle += math.pi * np.int32(angle < 0)
    return output, angle

## iterator
def iterate_regions(img, kernel_size):
    h, w, d = img.shape
    for i in range(h - kernel_size + 1):
        for j in range(w - kernel_size + 1):
            img_region = img[i:(i + kernel_size), j:(j + kernel_size)]
            yield img_region, i, j
