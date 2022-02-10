import cv2
import numpy as np

'''Open a new window to show the image'''
img_org = cv2.imread("./Dataset_OpenCvDl_Hw1/Q1_Image/Sun.jpg")

def load_image():
    cv2.imshow('Hw1-1',img_org)
    # 按下任意鍵則關閉所有視窗
    cv2.waitKey(0)

def color_seperation():
    '''Extract 3 channels of the image BGR to 3 separated channels and show the result images'''
    B, G, R = cv2.split(img_org)            # B, G, R各為單通道圖像（387, 620, 1)

    # 把B, G, R各自擴展回三通道圖像。方法：有值的放值，沒有值的地方就補0
    zeros = np.zeros(img_org.shape[:2], dtype="uint8")  # 生成一個值爲0的單通道數組
    # Way1
    cv2.imshow("B channel", cv2.merge([B, zeros, zeros]))
    cv2.imshow("G channel", cv2.merge([zeros, G, zeros]))
    cv2.imshow("R channel", cv2.merge([zeros, zeros, R]))
    cv2.waitKey(0)
    '''
    # Way2
    img_B = cv2.merge([B, zeros, zeros])
    img_G = cv2.merge([zeros, G, zeros])
    img_R = cv2.merge([zeros, zeros, R])
    COM = np.concatenate((img_B, img_G, img_R), axis=1)
    cv2.imshow('Hw1-2', COM)
    cv2.waitKey(0)
    '''

def color_transformation():
    '''Transform the image into grayscale image I1 by calling OpenCV function directly.'''
    img_gray_1 = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    cv2.imshow('I1',img_gray_1)
    cv2.waitKey(0)

    '''Merge BGR separated channel images from problem 1.2 into grayscale image I2 by I2 = (R+G+B)/3.'''
    B, G, R = cv2.split(img_org)        #B, G, R各為單通道圖像(387, 620, 1)
    img_gray_2 = np.zeros((387, 620), dtype = "uint8")
    for i in B:
        for j in B:
            img_gray_2[i][j] = np.round(B[i][j]/3 + G[i][j]/3 + R[i][j]/3)

    cv2.imshow('I2',img_gray_2)
    cv2.waitKey(0)

def blending():
    '''
    OpenCV影像相加：
    addWeighted([InputArray] src1, [double] alpha, [InputArray] src2, [double] beta, [double] gamma)
    src1：輸入圖。
    alpha：src1的權重。
    src2：輸入圖，和src1的尺寸和通道數相同。
    beta：src2的權重。
    gamma：兩圖相加後再增加的值。

    '''

    '''
    cv2.createTrackbar('R','image',0,255,Change_color)

    *'R' 為 Trackbar(軌道桿)顯示的名稱
    *'image'為 Trackbar(軌道桿)要依附在哪一個windows 視窗,在這裡為名稱'image'的視窗
    *0,255 為拉動桿子的數值變化範圍,在這個範例設定最小0 ,最大為255
    *Change_color 為拉動桿子的數值變化後產生事件要執行的副程式,在這個範例副程式為 def Change_color(x):

    '''

    # [load]
    src1 = cv2.imread("./Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Strong.jpg")
    src2 = cv2.imread("./Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Weak.jpg")

    # Create a black image, a window
    cv2.namedWindow('Blend', cv2.WINDOW_NORMAL)

    # [blend_images]
    def Blending_ajustment(x):
        alpha = x / 255
        beta = (1.0 - alpha)
        dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
        # [display]
        cv2.imshow('Blend', dst)

    cv2.createTrackbar('Blend', 'Blend', 0, 255, Blending_ajustment)
    cv2.waitKey(0)


cv2.destroyAllWindows()