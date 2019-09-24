from tkinter import *
from PIL import Image, ImageTk
import math
import sys
import cv2
import numpy as np
import scipy.stats as st
import os
import math

camera = cv2.VideoCapture(0)   # open camera

camera.set(4, 1)
root = Tk()
root.title("Camera1")


def video_loop():
    success, img = camera.read()   # read image from camera
    if success:
        cv2.waitKey(2)
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)   # Transfer color from RGB to RGBA
        current_image = Image.fromarray(cv2image)    # Transfer image to image object
        imgtk = ImageTk.PhotoImage(image=current_image)
        panel.imgtk = imgtk
        panel.config(image=imgtk)
        root.after(1, video_loop)


panel = Label(root)  # initialize image panel
panel.pack(padx=10, pady=10)
root.config(cursor="arrow")
btn1 = Button(root, text="capture image1", command=lambda : cap_tk1(cap=camera))
btn1.pack(fill="both", expand=False, padx=10, pady=10)
btn2 = Button(root, text="capture image2", command=lambda : cap_tk2(cap=camera))
btn2.pack(fill="both", expand=False, padx=10, pady=10)
btn3 = Button(root, text="Start", command=lambda : main())
btn3.pack(fill="both", expand=False, padx=10, pady=10)


def cap_tk1(cap):
    ret, frame = cap.read()
    cv2.imshow("capture1", frame)
    cv2.imwrite("original1.jpg", frame)


def cap_tk2(cap):
    ret, frame = cap.read()
    cv2.imshow("capture2", frame)
    cv2.imwrite("original2.jpg", frame)


file_1 = "colorGrid.png"
file_2 = "Grid.jpg"
image1 = cv2.imread(file_1)
image2 = cv2.imread(file_2)
gray_img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


def load_images():
    global image1
    global image2

    while image1.shape[0] > 750 or image1.shape[1] > 1200:
        image1 = cv2.resize(image1,(int(image1.shape[1]/2), int(image1.shape[0]/2)))
    image1 = gray_img1
    while image2.shape[0] > 750 or image2.shape[1] > 1200:
        image2 = cv2.resize(image2,(int(image2.shape[1]/2), int(image2.shape[0]/2)))
    image2 = gray_img2
    return image1, image2


def reloadimage():
    global image1
    global image2

    while image1.shape[0] > 750 or image1.shape[1] > 1200:
        image1 = cv2.resize(image1,(int(image1.shape[1]/2), int(image1.shape[0]/2)))
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)

    while image2.shape[0] > 750 or image2.shape[1] > 1200:
        image2 = cv2.resize(image2,(int(image2.shape[1]/2), int(image2.shape[0]/2)))

    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    return image1, image2


def nothing(x):
    pass



def gradient_x(h):
    delta_h = np.zeros(h.shape)
    a = int(h.shape[0])
    b = int(h.shape[1])

    for i in range(a):
        for j in range(b):
            if i - 1 >= 0 and i + 1 < a and j - 1 >= 0 and j + 1 < b:
                c = abs(int(h[i - 1, j - 1]) - int(h[i + 1, j - 1]) + 2 * (int(h[i - 1, j]) - int(h[i + 1, j])) + int(
                    h[i - 1, j + 1]) - int(h[i + 1, j + 1]))

                if c > 255:
                    c = 255
                delta_h[i, j] = c
            else:
                delta_h[i, j] = 0

    return delta_h


def gradient_y(h):
    delta_h = np.zeros(h.shape)
    a = int(h.shape[0])
    b = int(h.shape[1])
    for i in range(a):
        for j in range(b):
            if i - 1 >= 0 and i + 1 < a and j - 1 >= 0 and j + 1 < b:
                c = abs(int(h[i - 1, j - 1]) - int(h[i - 1, j + 1]) + 2 * (int(h[i, j - 1]) - int(h[i, j + 1])) + (
                            int(h[i + 1, j - 1]) - int(h[i + 1, j + 1])))  # 注意像素不能直接计算，需要转化为整型
                # print c
                if c > 255:
                    c = 255
                delta_h[i, j] = c
            else:
                delta_h[i, j] = 0

    return delta_h


def blocksize_handle():
    bs1 = cv2.getTrackbarPos('blockSize', 'Parameters for cornerHarris 1')
    if bs1 % 2 == 0:
        bs1 = bs1 + 1

    bs2 = cv2.getTrackbarPos('blockSize', 'Parameters for cornerHarris 2')
    if bs2 % 2 == 0:
        bs2 = bs2 + 1
    return bs1, bs2


def harris(gray_imag, blockSize, Gaussian, threshold):
    cv2.Laplacian(gray_imag, cv2.CV_64F, ksize=blockSize)

    dx = np.array(gradient_x(gray_imag))
    dy = np.array(gradient_y(gray_imag))

    Ixx = dx * dx
    Iyy = dy * dy
    Ixy = dx * dy

    Ixx = cv2.GaussianBlur(Ixx, (Gaussian, Gaussian), 1.5)
    Iyy = cv2.GaussianBlur(Iyy, (Gaussian, Gaussian), 1.5)
    Ixy = cv2.GaussianBlur(Ixy, (Gaussian, Gaussian), 1.5)

    a = int(gray_imag.shape[0])
    b = int(gray_imag.shape[1])
    R = np.zeros(gray_imag.shape)
    for i in range(a):
        for j in range(b):
            M = [[Ixx[i, j], Ixy[i, j]], [Ixy[i, j], Iyy[i, j]]]
            R[i, j] = np.linalg.det(M) - threshold * (np.trace(M)) * (np.trace(M))

    return R

def main():
    # Creat trackbar for img1cv2.WINDOW_KEEPRATIO
    cv2.namedWindow('Parameters for cornerHarris 1', cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow("Parameters for cornerHarris 1", 300, 300);
    cv2.createTrackbar('blockSize', 'Parameters for cornerHarris 1', 3, 10, nothing)
    cv2.createTrackbar('Gaussian(odd)', 'Parameters for cornerHarris 1', 1, 15, nothing)
    cv2.createTrackbar('thres*100', 'Parameters for cornerHarris 1', 1, 6, nothing)
    cv2.createTrackbar('traceWeight', 'Parameters for cornerHarris 1', 1, 3, nothing)

    # Creat trackbar for img2
    cv2.namedWindow('Parameters for cornerHarris 2', cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow("Parameters for cornerHarris 2", 300, 300);
    cv2.createTrackbar('blockSize', 'Parameters for cornerHarris 2', 3, 10, nothing)
    cv2.createTrackbar('Gaussian(odd)', 'Parameters for cornerHarris 2', 1, 29, nothing)
    cv2.createTrackbar('thres*100', 'Parameters for cornerHarris 2', 1, 10, nothing)
    cv2.createTrackbar('traceWeight', 'Parameters for cornerHarris 2', 1, 3, nothing)
    while (True):
        # user press ESC break the programm
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # user press the h key, the help information will show
        if k == 72:
            print("This program is used to detect the corner by applying harris algorithm.")

        # for show the result
        img1, img2 = load_images()

        blocksize1, blocksize2 = blocksize_handle()
        # get the Gaussian param. fromm trackbar
        Gaussian1 = cv2.getTrackbarPos('Gaussian(odd)', 'Parameters for cornerHarris 1')
        if Gaussian1 % 2 == 0:
            Gaussian1 = Gaussian1 + 1

            # get the Gaussian param. fromm trackbar
        Gaussian2 = cv2.getTrackbarPos('Gaussian(odd)', 'Parameters for cornerHarris 2')
        if Gaussian2 % 2 == 0:
            Gaussian2 = Gaussian2 + 1

        # get the threshold param. fromm trackbar,the real threshol = threshold1*dst1.max()
        threshold1 = float(cv2.getTrackbarPos('thres*100', 'Parameters for cornerHarris 1') / 100)
        threshold2 = float(cv2.getTrackbarPos('thres*100', 'Parameters for cornerHarris 2') / 100)
        # the trace weigth for corner detector
        traceWeight1 = cv2.getTrackbarPos('traceWeight', 'Parameters for cornerHarris 1')
        traceWeight2 = cv2.getTrackbarPos('traceWeight', 'Parameters for cornerHarris 2')
        Harris_detector1 = harris(gray_img1, blocksize1, Gaussian1, threshold1)
        Harris_detector2 = harris(gray_img2, blocksize2, Gaussian2, threshold2)
        # result is dilated for marking the corners
        dst1 = cv2.dilate(Harris_detector1, None)
        dst2 = cv2.dilate(Harris_detector2, None)

        # Threshold for an optimal value, it may vary depending on the image.
        # and translate into Binary image

        ret1, dst1 = cv2.threshold(dst1, threshold1 * dst1.max(), 255, cv2.THRESH_BINARY)
        ret2, dst2 = cv2.threshold(dst2, threshold1 * dst2.max(), 255, cv2.THRESH_BINARY)

        dst1 = np.uint8(dst1)
        dst2 = np.uint8(dst2)

        # To find the centroid of every corner detected point,and draw an empty rectangle.
        ret1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(dst1)
        ret2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(dst2)

        for i in centroids1:
            # print ('i=',i)
            # the size of rectangle is 16*16
            x1 = int(i[0] - 5)
            y1 = int(i[1] - 5)
            x2 = int(i[0] + 5)
            y2 = int(i[1] + 5)
            # draw an empty rectangle
            cv2.rectangle(img1, (x1, y1), (x2, y2), (255, 255, 0), traceWeight1)

        for i in centroids2:
            # print ('i=',i)
            # the size of rectangle is 16*16
            x1 = int(i[0] - 5)
            y1 = int(i[1] - 5)
            x2 = int(i[0] + 5)
            y2 = int(i[1] + 5)
            # draw an empty rectangle
            cv2.rectangle(img2, (x1, y1), (x2, y2), (255, 255, 0), traceWeight2)

            # show the result
        cv2.imshow('Parameters for cornerHarris 1', img1)
        cv2.waitKey(2)
        cv2.imshow('Parameters for cornerHarris 2', img2)

        cv2.waitKey(2)

    cv2.destroyAllWindows()

video_loop()
root.mainloop()
camera.release()
