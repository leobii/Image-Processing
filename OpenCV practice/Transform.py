import cv2
import numpy as np

img_org = cv2.imread("./Dataset_OpenCvDl_Hw1/Q4_Image/SQUARE-01.png")
height, width = 256, 256
#height = int(input("Please enter the height："))
#width = int(input("Please enter the width: "))

img_resize = cv2.resize(img_org,(width,height))

'''pre-Translation'''
tx = 0
ty = 60

def resize():

    cv2.imshow('img',img_resize)
    # 按下任意鍵則關閉所有視窗
    cv2.waitKey(0)

def translation():

    translation_matrix = np.array([[1, 0, tx],[0, 1, ty]], dtype=np.float32)
    # apply the translation to the image
    # wrapaffine是開一張畫布
    # src是放進畫布裡面的圖片
    # M是你要對SRC做的事情
    # dsize是設定畫布大小
    img_trans = cv2.warpAffine(src=img_resize, M=translation_matrix, dsize=(width*2, height*2))

    cv2.imshow('img_2', img_trans)
    # 按下任意鍵則關閉所有視窗
    cv2.waitKey(0)

def rotation():
    # get the center coordinates of the image to create the 2D rotation matrix
    center = ((width + tx) / 2 , (height + ty) / 2)
    # apply the rotation to the image
    rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=10, scale=0.5)
    img_rot = cv2.warpAffine(src=img_resize, M=rotation_matrix, dsize=(400,300))
    cv2.imshow('img_3', img_rot)
    # 按下任意鍵則關閉所有視窗
    cv2.waitKey(0)

def shearing():
    # get the image shape
    rows, cols, dim = img_resize.shape
    # transformation matrix for Shearing
    # shearing applied to x-axis
    #M = np.float32([[1, 0.5, 0],
    #                [0, 1  , 0],
    #                [0, 0  , 1]])
    # shearing applied to y-axis
    # M = np.float32([[1,   0, 0],
    #             	  [0.5, 1, 0],
    #             	  [0,   0, 1]])
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    sheared_img = cv2.warpAffine(img_resize, M, (cols, rows))

    # show the resulting image
    cv2.imshow('img-4',sheared_img)
    # 按下任意鍵則關閉所有視窗
    cv2.waitKey(0)


cv2.destroyAllWindows()




