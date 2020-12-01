import numpy as np
import cv2 as cv
import math


def dist_squared(point1, point2):
    return (point2[1] - point1[1]) ** 2 + (point2[0] - point1[0]) ** 2


def ratio_fits(width, height):
    max_ratio = 3
    min_ratio = 1

    r = max(width, height) / min(width, height)
    return min_ratio < r < max_ratio


def update(val):
    global img
    global imgGray
    global canvas

    global minSignSize
    global maxSignSize

    thresh1 = cv.getTrackbarPos("thresh1", "pars")
    if thresh1 % 2 == 0:
        thresh1 += 1

    thresh = cv.adaptiveThreshold(imgGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, thresh1, 2)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    canvas = img.copy()

    for contour in contours:

        if len(contour) < 40:
            cv.drawContours(canvas, [contour], -1, (0, 0, 0), 1)
            continue

        x, y, w, h = cv.boundingRect(contour)

        if w < minSignSize or h < minSignSize:
            cv.drawContours(canvas, [contour], -1, (255, 0, 0), 1)
            continue

        rect = cv.minAreaRect(contour)

        box = cv.boxPoints(rect)
        box = np.int0(box)

        max_size_squared = maxSignSize ** 2
        width_squared = dist_squared(box[0], box[1])
        height_squared = dist_squared(box[1], box[2])

        if width_squared > max_size_squared and height_squared > max_size_squared:
            cv.drawContours(canvas, [contour], -1, (0, 0, 255), 1)
            continue

        width = math.sqrt(width_squared)
        height = math.sqrt(height_squared)

        if not ratio_fits(width, height):
            cv.drawContours(canvas, [contour], -1, (0, 0, 0), 2)
            continue

        area = cv.contourArea(contour)
        min_bounding_take_up = 0.5

        if area / (width * height) < min_bounding_take_up:
            cv.drawContours(canvas, [contour], -1, (255, 255, 255), 1)
            continue

        epsilon = 0.01 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)

        cv.drawContours(canvas, [box], 0, (255, 0, 255), 2)
        cv.drawContours(canvas, [approx], -1, (0, 255, 0), 2)


cv.namedWindow("pars")
cv.resizeWindow("pars", 640, 240)
cv.createTrackbar("thresh1", "pars", 11, 30, update)
cv.createTrackbar("thresh2", "pars", 20, 100, update)

kernelSize = 5
sigma = 3

img = cv.imread('./res/test05.png')
imgBlur = cv.GaussianBlur(img, (kernelSize, kernelSize), cv.BORDER_DEFAULT)
imgGray = cv.cvtColor(imgBlur, cv.COLOR_BGR2GRAY)

imgWidth = img.shape[0]
imgHeight = img.shape[1]
imgMinExtent = min(imgWidth, imgHeight)

minSignSize = imgMinExtent * 0.03
maxSignSize = imgMinExtent * 0.20
print(minSignSize, maxSignSize)

canvas = img.copy()
update(None)

cv.imshow('image', canvas)

while cv.getWindowProperty('image', 0) >= 0:
    cv.imshow('image', canvas)
    cv.waitKey(500)
