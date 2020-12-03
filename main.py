import numpy as np
import cv2 as cv

def get_aspect_ratio(rect):
	return rect[1][1] / rect[1][0]


def get_greater_aspect_ratio(rect):
	width = rect[1][0]
	height = rect[1][1]
	return (width / height) if width > height else (height / width)


def ratio_fits(rect):
	min_ratio = 1.3
	max_ratio = 3.2

	ratio = get_greater_aspect_ratio(rect)
	return min_ratio < ratio < max_ratio


class SetShape:

	def __init__(self, contour, bbox, exact_rect, shape):
		self.contour = contour
		self.bbox = bbox
		self.exact_rect = exact_rect
		self.shape = shape

def update(val):
	global imgGray

	min_contour_points = imgMinExtent * 0.05

	min_bounds_size = imgMinExtent * 0.03
	min_exact_bound_size = imgMinExtent * 0.03
	max_exact_bound_size = imgMinExtent * 0.19

	min_bounds_occupation = 0.5
	max_bounds_occupation = 0.90

	max_diamond_occupation = 0.65
	max_squiggle_occupation = 0.81

	thresh1 = cv.getTrackbarPos("thresh1", "pars")

	if thresh1 % 2 == 0:
		thresh1 += 1

	thresh = cv.adaptiveThreshold(imgGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, thresh1, 2)

	contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

	matched_shapes = []

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

		shape = "diamond" if bounds_occupation < max_diamond_occupation else \
				"squiggle" if bounds_occupation < max_squiggle_occupation else \
				"oval"

		set_shape = SetShape(contour, [x, y, w, h], rect, shape)
		matched_shapes.append(set_shape)

	find_cards(matched_shapes)


def get_shape_color(shape):
	return {
		"diamond": (255, 0, 0),
		"squiggle": (0, 255, 0),
		"oval": (0, 0, 255)
	}[shape]


def dist_squared(point0, point1):
	return (point1[0] - point0[0]) ** 2 + (point1[1] - point0[1])


def find_cards(shapes):

	global img
	global canvas

	unique_shapes = []
	contained = []

	for shape in shapes:

		if shape in unique_shapes:
			continue

		is_unique = True

		for other in shapes:

			if shape is other:
				continue

			point = shape.contour[0][0]

			if cv.pointPolygonTest(other.contour, (point[0], point[1]), False) > 0:
				contained.append(shape)
				is_unique = False
				break

		if is_unique is True:
			unique_shapes.append(shape)

	canvas = img.copy()

	for shape in unique_shapes:

		box = cv.boxPoints(shape.exact_rect)
		box = np.int0(box)

		# hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

		mask = np.zeros(img.shape[:2], np.uint8)
		cv.drawContours(mask, [shape.contour], -1, 255, -1)
		mean = cv.mean(img, mask)

		cv.drawContours(canvas, [box], -1, (255, 0, 255), 2)
		cv.drawContours(canvas, [shape.contour], -1, (int(mean[0]), int(mean[1]), int(mean[2])), 2)


cv.namedWindow("pars")
cv.resizeWindow("pars", 640, 240)
cv.createTrackbar("thresh1", "pars", 9, 30, update)
cv.createTrackbar("thresh2", "pars", 20, 100, update)

kernelSize = 5
sigma = 3

img = cv.imread('./res/test07.JPG')

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
