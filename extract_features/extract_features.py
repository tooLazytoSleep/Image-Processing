import cv2
import numpy as np
import sys
def gray_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray
def extractFeats(image):
    num = 30
    thel = 0.001
    # termination criteria
    criteria_eps = cv2.TERM_CRITERIA_EPS
    criteria_max = cv2.TERM_CRITERIA_MAX_ITER
    criteria = (criteria_eps + criteria_max, num, thel)
    w = 6
    h = 9
    dim = 3
    objp = np.zeros((w*h, dim), np.float32)
    objp[:,:2] = np.mgrid[0:h,0:w].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    gray = gray_img(image)
    # Find the chess board corners
    row = 9
    col = 6
    ret, corners = cv2.findChessboardCorners(gray, (row, col), None)
    # If found, add object points, image points (after refining them)
    if ret:
        corners2=cv2.cornerSubPix(gray,corners, (11, 11), (-1, -1), criteria)
        # Draw and display the corners
        cv2.drawChessboardCorners(image, (row,col), corners2, ret)
        cv2.imshow("image", image)
        file = open("data/points_correspondence.txt", "w")
        write_file(objp, corners, file)
        file.close()
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_file(objp, corners, file):
    for i, j in zip(objp, corners.reshape(-1, 2)):
        file.write(str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + ' ' + str(j[0]) + ' ' + str(j[1]) + '\n')

def main():
    image = cv2.imread("data/chessboard.jpg")
    extractFeats(image)
