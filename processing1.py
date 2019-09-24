from tkinter import *
import cv2
from PIL import Image, ImageTk
import numpy as np
from scipy import ndimage as ni
import math


camera = cv2.VideoCapture(0)   # open camera
root = Tk()
root.title("Camera")


def video_loop():
    success, img = camera.read()   # read image from camera
    if success:
        cv2.waitKey(0)
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)   # Transfer color from RGB to RGBA
        current_image = Image.fromarray(cv2image)    # Transfer image to image object
        imgtk = ImageTk.PhotoImage(image=current_image)
        panel.imgtk = imgtk
        panel.config(image=imgtk)
        root.after(1, video_loop)


panel = Label(root)  # initialize image panel
panel.pack(padx=10, pady=10)
root.config(cursor="arrow")
btn = Button(root, text="capture", command=lambda : cap_tk(cap=camera))
btn.pack(fill="both", expand=False, padx=10, pady=10)


def cap_tk(cap):
    ret, frame = cap.read()
    cv2.imshow("capture", frame)
    cv2.imwrite("original.jpg", frame)
    key()


def capture():
    global image
    image = cv2.imread("original.jpg")
    cv2.imshow("capture", image)
    return image


def img2gray_cv(image):
    img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return img


def img2gray_self(image):
    row, col, channel = image.shape
    image_gray = np.zeros((row, col))
    for r in range(row):
        for l in range(col):
            image_gray[r, l] = 1 / 3 * image[r, l, 0] + 1 / 3 * image[r, l, 1] + 1 / 3 * image[r, l, 2]
    return image_gray


def switch_image(img,count):
    count = count%3
    if len(image.shape) == 3:
        if count == 0:
            image[:, :, 1] = 0
            image[:, :, 2] = 0
        elif count == 1:
            image[:, :, 0] = 0
            image[:, :, 2] = 0
        else:
            image[:, :, 0] = 0
            image[:, :, 1] = 0
        cv2.imshow("capture", image)
    else:
        print("Not possible to convert to b, g or r")


def smooth_cv(n):
    global image
    image = capture()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((n, n), np.float32) / (n * n)
    image = cv2.filter2D(image, -1, kernel)
    cv2.imshow('capture', image)


def smooth_self(n):
    img = ni.gaussian_filter(image, n, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
    cv2.imshow("capture", img)


def sobel_func(x, y, img):
    derivative = cv2.Sobel(img, cv2.CV_64F, x, y, ksize=3)
    return derivative

def vectors_handle(n):
    global image
    image = capture()
    gray_img = img2gray_cv(image)
    derivative_x = sobel_func(1, 0, gray_img)
    derivative_y = sobel_func(0, 1, gray_img)
    rows = gray_img.shape[0]
    cols = gray_img.shape[1]
    if n != 0:
        for x in range(0, rows, n):
            for y in range(0, cols, n):
                theta = math.atan2(derivative_y[x, y], derivative_x[x, y])
                theta_x = x + n * math.cos(theta)
                theta_y = y + n * math.sin(theta)
                grad_x = int(theta_x)
                grad_y = int(theta_y)
                cv2.arrowedLine(gray_img, (y, x), (grad_y, grad_x), (0, 0, 0))
    cv2.imshow('capture', gray_img)


def rotate(n):
    rot = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), n, 1)
    img = cv2.warpAffine(image, rot, (image.shape[1], image.shape[0]))
    cv2.imshow("capture", img)


def description():
        print("'i': Reloading the original image.")
        print("'w': Saving the current image in to 'out.jpg'.")
        print("'g': Converting the image to grayscale using the openCV conversion function.")
        print("'G': Converting the image to grayscale using myself implementation.")
        print("'c': Cycling through the color channels")
        print("'s': Converting the image to grayscale and smooth using the openCV function.")
        print("'S': Converting the image to grayscale and smooth using my function.")
        print("'d': Downsampling the image by a factor of 2 without smoothing.")
        print("'D': Downsampling the image by a factor of 2 with smoothing.")
        print("'x': Converting the image to grayscale and perform  convolution with a x derivative filter.")
        print("'y': Converting the image to grayscale and perform  convolution with a y derivative filter.")
        print("'m': Showing the magnitude of the gradient normalized to the range [0, 255].")
        print("'p':Converting the image to grayscale and plot the gradient vector of the image every N pixels and let the plotted gradient vector have a length of K.")
        print("'r': Converting the image to grayscale and rotate it.\n")


c = 0
def key():
    global image,c
    while True:
        key_down = cv2.waitKey(0)
        if key_down == ord("i"):
            print("Reloading the original image...")
            cv2.destroyAllWindows()
            image = capture()
            cv2.imshow("capture", image)
            print("Reloading the original image successfully!")
        elif key_down == ord("w"):
            print("Saving the current image in to 'out.jpg'...")
            cv2.imwrite("out.jpg", image)
            print("Saving the current image in to 'out.jpg' successfully!")
        elif key_down == ord("g"):
            print("Converting the image to grayscale using the openCV conversion function...")
            capture()
            img=img2gray_cv(image)
            cv2.imshow("capture", img)
            print("Converting the image to grayscale using the openCV conversion function successfully!")
        elif key_down == ord("G"):
            print("Converting the image to grayscale using my implementation...")
            capture()
            img = img2gray_self(image)
            cv2.imshow("capture", img.astype("uint8"))
            print("Converting the image to grayscale using my implementation successfully!")
        elif key_down == ord("c"):
            print("Cycling through the color channels...")
            cv2.destroyAllWindows()
            capture()
            c = c+1
            switch_image(image, c)
            print("Cycling through the color channels successfully!")
        elif key_down == ord("s"):
            print("Converting the image to grayscale and smooth using the openCV function...")
            cv2.destroyAllWindows()
            capture()
            print("Converting the image to grayscale and smooth using the openCV function successfully!")
            cv2.createTrackbar('smoothing', "capture", 0, 20, smooth_cv)
        elif key_down == ord("S"):
            print("Converting the image to grayscale and smooth using my function...")
            cv2.destroyAllWindows()
            capture()
            print("Converting the image to grayscale and smooth using my function successfully!")
            cv2.createTrackbar('smoothing', "capture", 0, 20, smooth_self)
        elif key_down == ord("d"):
            print("Downsampling the image by a factor of 2 without smoothing...")
            cv2.destroyAllWindows()
            capture()
            resize_img = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))
            cv2.imshow("capture", resize_img)
            print("Downsampling the image by a factor of 2 without smoothing successfully!")
        elif key_down == ord("D"):
            print("downsampling the image by a factor of 2 with smoothing...")
            cv2.destroyAllWindows()
            capture()
            resize_img = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))
            smooth_img = cv2.GaussianBlur(resize_img, (5, 5), 0)
            cv2.imshow("capture", smooth_img)
            print("downsampling the image by a factor of 2 with smoothing successfully!")
        elif key_down == ord("x"):
            print("Converting the image to grayscale and perform  convolution with a x derivative filter...")
            cv2.destroyAllWindows()
            capture()
            gray_img = img2gray_cv(image)
            sobel_img = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
            img = cv2.normalize(sobel_img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            cv2.imshow("capture", img)
            print("Converting the image to grayscale and perform  convolution with a x derivative filter successfully!")
        elif key_down == ord("y"):
            print("Converting the image to grayscale and perform  convolution with a y derivative filter...")
            cv2.destroyAllWindows()
            capture()
            gray_img = img2gray_cv(image)
            sobel_img = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1)
            img = cv2.normalize(sobel_img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            cv2.imshow("capture", img)
            print("Converting the image to grayscale and perform  convolution with a y derivative filter successfully!")
        elif key_down == ord("m"):
            print("Loading the magnitude of the gradient normalized to the range [0, 255]...")
            cv2.destroyAllWindows()
            capture()
            gray_img = img2gray_cv(image)
            derivative_x = sobel_func(1, 0, gray_img)
            derivative_y = sobel_func(0, 1, gray_img)
            gradient = cv2.magnitude(derivative_x, derivative_y)
            img = cv2.normalize(gradient, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            cv2.imshow("capture", img)
            print("Loading the magnitude of the gradient normalized to the range [0, 255] successfully!")
        elif key_down == ord("p"):
            print("Converting the image to grayscale and plot the gradient vector of the image every N pixels and let the plotted gradient vector have a length of K...")
            cv2.destroyAllWindows()
            capture()
            cv2.createTrackbar('vector', 'capture', 0, 255, vectors_handle)
            print("Converting the image to grayscale and plot the gradient vector of the image every N pixels and let the plotted gradient vector have a length of K successfully!")
        elif key_down == ord("r"):
            print("Converting the image to grayscale and rotate it...")
            cv2.destroyAllWindows()
            capture()
            cv2.imshow("capture", image)
            cv2.createTrackbar('rotation', "capture", 0, 360, rotate)
            print("Converting the image to grayscale and rotate it successfully!")
        elif key_down == ord("h"):
            print("Printing the description...")
            description()
            print("Printing the description successfully!")


cv2.destroyAllWindows()
video_loop()
root.mainloop()
camera.release()
