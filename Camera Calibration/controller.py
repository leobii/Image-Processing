from PyQt5 import QtWidgets, QtGui, QtCore
from UI import Ui_MainWindow

'''''''''''''''''
Camera Calibration
'''''''''''''''''
import numpy as np
import cv2 as cv
import glob

#棋盤格模板規格(格子和格子中間的點點，不是有幾個格子)
corners_vertical = 8
corners_horizontal = 11
pattern_size = (corners_horizontal, corners_vertical)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((corners_horizontal*corners_vertical,3), np.float32)
objp[:,:2] = np.mgrid[0:corners_horizontal,0:corners_vertical].T.reshape(-1,2)
#print(objp)
# Arrays to store object points and image points from all the images.
objpoints = []       # 3d point in real world space (世界坐標系)
imgpoints = []       # 2d points in image plane. (其對應的圖像點)
images = glob.glob('./Dataset_OpenCvDl_Hw2/Q2_Image/*.bmp')


class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow_controller, self).__init__()          # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        # TODO
        self.ui.Find_Corners.clicked.connect(self.find_corners)
        self.ui.Find_Intrinsic_Matric.clicked.connect(self.find_intrinsic)
        self.ui.pushButton_2.clicked.connect(self.find_extrinsic)
        self.ui.Find_Distortion.clicked.connect(self.find_distortion)
        self.ui.Show_Results.clicked.connect(self.show_results)

    def find_corners(self):

        for fname in images:
            img = cv.imread(fname)
            img_resize = cv.resize(img, (600, 600))  # 設定大小
            gray = cv.cvtColor(img_resize, cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            # return_value(返回值), corners = cv2.findChessboardCorners(image, patternSize, corners, flags)
            ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

            # If found(return_value == True), add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # increase the accuracy of corners by using 'cv.cornerSubPix'
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # Draw and display the corners
                # cv2.drawChessboardCorners(image, patternSize, corners, patternWasFound)
                # image：目標圖像。它必須是8位灰度或彩色圖像
                # patternSize：每個棋盤行和列的內角數
                # corners：輸出檢測到的角點陣列
                # patternWasFound：參數指示是否找到完整的板。 findChessboardCorners的return value。
                cv.drawChessboardCorners(img_resize, pattern_size, corners2, ret)

                cv.imshow('Find Corners', img_resize)
                cv.waitKey(500)

        cv.destroyAllWindows()
        self.ret, self.In_mtx, self.distortion, self.rotation_vecs, self.translation_vecs = cv.calibrateCamera(objpoints, imgpoints,
                                                                                      gray.shape[::-1], None, None)

    def find_intrinsic(self):
        print("Intrinsic Matrix(內參數矩陣): \n", self.In_mtx)

    def find_extrinsic(self):
        select_image_num = int(self.ui.lineEdit.text())
        if select_image_num>=0 & select_image_num<=14:
            print(int(select_image_num))

            rotation_ars = np.array(self.rotation_vecs)
            traslation_ars = np.array(self.translation_vecs)

            # ATTENTION!!! We need to change Vectors into Matrix!
            # Using cv.Rodrigues.
            # Input: rotation vector (3x1 or 1x3) or rotation matrix (3x3).
            # Output: rotation matrix (3x3) or rotation vector (3x1 or 1x3), respectively;
            #         and Jacobian matrix (3x9 or 9x3).
            rotation_mtx = np.zeros(shape=(3, 3))
            rotation_mtx,_ = cv.Rodrigues(rotation_ars[select_image_num,0:3,0])

            print(rotation_mtx)
            print(traslation_ars[select_image_num])
            Ex_mtx = np.hstack([rotation_mtx, traslation_ars[select_image_num]])
            print(Ex_mtx)
        else:
            print("ERROR!")

    def find_distortion(self):
        print("Distortion(畸變係數): \n", self.distortion)

    def show_results(self):
        for fname in images:
            img = cv.imread(fname)
            img_resize = cv.resize(img, (600, 600))  # 設定大小
            h, w, _ = img_resize.shape  # img.shape[0:2]

            # Get new camera matrix
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.In_mtx, self.distortion, (w, h), 1, (w,h))
            # undistort
            dst = cv.undistort(img_resize, self.In_mtx, self.distortion, None, newcameramtx)
            # crop the image
            print(roi)
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]

            dst_resize = cv.resize(dst, (600, 600))

            numpy_horizontal = np.hstack((img_resize, dst_resize))

            cv.imshow('Undistorted & Distorted', numpy_horizontal)
            cv.waitKey(500)

        cv.destroyAllWindows()












