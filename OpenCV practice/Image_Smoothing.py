import cv2

'''Open a new window to show the image'''
img_org_whiteNoise = cv2.imread("./Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_whiteNoise.jpg")
img_org_pepperSalt = cv2.imread("./Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_pepperSalt.jpg")
# Ref: https://docs.opencv.org/3.4/d4/d13/tutorial_py_filtering.html

def gaussian_blur():
    cv2.imshow('Original(whiteNoise)', img_org_whiteNoise)
    img_gaussian_blur = cv2.GaussianBlur(img_org_whiteNoise,(5,5),0)
    cv2.imshow('Gaussain Blur', img_gaussian_blur)
    # 按下任意鍵則關閉所有視窗
    cv2.waitKey(0)

def bilateral_filter():
    cv2.imshow('Original(whiteNoise)', img_org_whiteNoise)
    img_bilateral_filter = cv2.bilateralFilter(img_org_whiteNoise, 9, 90, 90)
    cv2.imshow('Bilateral Filter', img_bilateral_filter)
    # 按下任意鍵則關閉所有視窗
    cv2.waitKey(0)

def median_filter():
    cv2.imshow('Original(pepperSalt)', img_org_pepperSalt)
    img_median_filter_kernelsize_3 = cv2.medianBlur(img_org_pepperSalt, 3)
    img_median_filter_kernelsize_5 = cv2.medianBlur(img_org_pepperSalt, 5)
    cv2.imshow('3x3 Median Filter', img_median_filter_kernelsize_3)
    cv2.imshow('5x5 Median Filter', img_median_filter_kernelsize_5)
    # 按下任意鍵則關閉所有視窗
    cv2.waitKey(0)