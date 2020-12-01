import numpy as np
import cv2 as cv
from copy import deepcopy


def dist_squared(point1, point2):
	return (point2[1] - point1[1]) ** 2 + (point2[0] - point1[0]) ** 2


def get_aspect_ratio(rect):
	return rect[1][1] / rect[1][0]


def get_greater_aspect_ratio(rect):
	width = rect[1][0]
	height = rect[1][1]
	return (width / height) if width > height else (height / width)


def ratio_fits(rect):
	min_ratio = 1.5
	max_ratio = 3.2

	ratio = get_greater_aspect_ratio(rect)
	return min_ratio < ratio < max_ratio


def get_contour_adjusted(rect, contour):
	center = rect[0]
	contour_norm = contour - center

	angle = -rect[2] + (90 if get_aspect_ratio(rect) > 1 else 0)
	angle_rad = np.radians(angle)

	matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
					   [np.sin(angle_rad), np.cos(angle_rad)]])

	contour_rot = np.empty((0, 1, 2), float)

	for point in contour_norm:
		contour_rot = np.append(contour_rot, [[matrix.dot(point[0])]], axis=0)

	rect2 = (deepcopy(rect[0]), deepcopy(rect[1]), rect[2] + angle)
	return np.int0(contour_rot + center), rect2


def update(val):
	global img
	global imgGray
	global canvas

	min_contour_points = imgMinExtent * 0.05

	min_bounds_size = imgMinExtent * 0.03
	min_exact_bound_size = imgMinExtent * 0.03
	max_exact_bound_size = imgMinExtent * 0.19

	min_bounds_occupation = 0.5
	max_bounds_occupation = 0.90

	max_diamond_occupation = 0.65
	max_wave_occupation = 0.81

	thresh1 = cv.getTrackbarPos("thresh1", "pars")

	if thresh1 % 2 == 0:
		thresh1 += 1

	thresh = cv.adaptiveThreshold(imgGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, thresh1, 2)

	contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	matched_shapes = []

	canvas = img.copy()

	for contour in contours:

		# ignore short contours
		if len(contour) < min_contour_points:
			# cv.drawContours(canvas, [contour], -1, (128, 128, 128), 1)
			continue

		x, y, w, h = cv.boundingRect(contour)

		# ignore small dots
		if w < min_bounds_size and h < min_bounds_size:
			cv.drawContours(canvas, [contour], -1, (255, 0, 0), 1)
			continue

		rect = cv.minAreaRect(contour)
		box = cv.boxPoints(rect)
		box = np.int0(box)

		width = rect[1][0]
		height = rect[1][1]

		# ignore quiet thing things
		if width < min_exact_bound_size or height < min_exact_bound_size:
			cv.drawContours(canvas, [contour], -1, (255, 128, 128), 1)
			continue

		if width > max_exact_bound_size and height > max_exact_bound_size:
			cv.drawContours(canvas, [contour], -1, (0, 0, 255), 1)
			continue

		if not ratio_fits(rect):
			cv.drawContours(canvas, [contour], -1, (0, 0, 0), 1)
			continue

		area = cv.contourArea(contour)
		bounds_occupation = area / (width * height)

		if bounds_occupation < min_bounds_occupation:
			cv.drawContours(canvas, [contour], -1, (255, 255, 255), 1)
			continue

		if bounds_occupation > max_bounds_occupation:
			cv.drawContours(canvas, [contour], -1, (255, 255, 255), 1)
			continue

		color = (255, 0, 0) if bounds_occupation < max_diamond_occupation \
			else (0, 255, 0) if bounds_occupation < max_wave_occupation \
			else (0, 0, 255)

		cv.drawContours(canvas, [contour], -1, color, 2)
		cv.drawContours(canvas, [box], 0, (255, 0, 255), 1)

		matched_shapes.append(contour)
		continue

	# new_contour, new_rect = get_contour_adjusted(rect, contour)
	# new_box = cv.boxPoints(new_rect)
	# new_box = np.int0(new_box)

	# point = matched_shapes[0][0][0]
	# point = tuple(map(tuple, point))
	# print(point)
	#
	# for shape in matched_shapes:
	# 	contained = cv.pointPolygonTest(shape, point, False)
	# 	cv.drawContours(canvas, [shape], -1, (0, 0, 255 if contained else 0), 2)
	#
	# cv.drawContours(canvas, [matched_shapes[0]], -1, (255, 0, 255), 2)


# cv.drawContours(canvas, [new_contour], -1, (0, 255, 0), 1)
# cv.drawContours(canvas, [new_box], 0, (255, 0, 255), 1)

cv.namedWindow("pars")
cv.resizeWindow("pars", 640, 240)
cv.createTrackbar("thresh1", "pars", 9, 30, update)
cv.createTrackbar("thresh2", "pars", 20, 100, update)

kernelSize = 5
sigma = 3

img = cv.imread('./res/test04.JPG')

if img is None:
	raise Exception("image not found")

imgBlur = cv.GaussianBlur(img, (kernelSize, kernelSize), cv.BORDER_DEFAULT)
imgGray = cv.cvtColor(imgBlur, cv.COLOR_BGR2GRAY)

imgWidth = img.shape[0]
imgHeight = img.shape[1]
imgMinExtent = min(imgWidth, imgHeight)

canvas = img.copy()
update(None)

cv.imshow('image', canvas)

while cv.getWindowProperty('image', 0) >= 0:
	cv.imshow('image', canvas)
	cv.waitKey(200)
