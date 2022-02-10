from PyQt5 import QtWidgets, QtGui, QtCore
from UI import Ui_MainWindow
from Image_Processing import load_image, color_seperation, color_transformation, blending
from Image_Smoothing import gaussian_blur, bilateral_filter, median_filter
from Edge_Detection import gaussian_blur_2, sobel_x, sobel_y, magnitude
from Transform import resize, translation, rotation, shearing


class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow_controller, self).__init__()          # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        # TODO
        self.clicked_counter = 0
        #
        self.ui.Load_Image.clicked.connect(self.buttonClicked)
        self.ui.Load_Image.clicked.connect(load_image)
        self.ui.Color_Seperation.clicked.connect(color_seperation)
        self.ui.Color_Transformation.clicked.connect(color_transformation)
        self.ui.Blending.clicked.connect(blending)

        # Image Smoothing
        self.ui.Gaussian_Blur.clicked.connect(gaussian_blur)
        self.ui.Bilatera_Filter.clicked.connect(bilateral_filter)
        self.ui.Median_Filter.clicked.connect(median_filter)

        # Edge Detection
        self.ui.Gaussian_Blur_2.clicked.connect(gaussian_blur_2)
        self.ui.Sobel_X.clicked.connect(sobel_x)
        self.ui.Sobel_Y.clicked.connect(sobel_y)
        self.ui.Magnitude.clicked.connect(magnitude)

        # Transform
        self.ui.Resize.clicked.connect(resize)
        self.ui.Translation.clicked.connect(translation)
        self.ui.Rotation_and_Scaling.clicked.connect(rotation)
        self.ui.Shearing.clicked.connect(shearing)

    def buttonClicked(self):
        self.clicked_counter += 1
        print(f"You clicked {self.clicked_counter} times.")




