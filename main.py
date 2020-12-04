import numpy as np
import cv2 as cv
import colorsys


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

	def __init__(self, contour, bbox, exact_rect, shape, color=None, filling=None, child_contour=None,
				 parent_contour=None):
		self.contour = contour
		self.bbox = bbox
		self.exact_rect = exact_rect
		self.shape = shape
		self.color = color
		self.filling = filling
		self.child_contour = child_contour
		self.parent_contour = parent_contour
		self.max_extent = max(exact_rect[1][0], exact_rect[1][0])


def update(val):
	global imgGray

	min_contour_points = imgMinExtent * 0.04
	min_bounds_size = imgMinExtent * 0.03
	min_exact_bound_size = imgMinExtent * 0.025
	max_exact_bound_size = imgMinExtent * 0.19

	min_bounds_occupation = 0.55
	max_bounds_occupation = 0.90

	max_diamond_occupation = 0.65
	max_squiggle_occupation = 0.81

	thresh1 = 9  # cv.getTrackbarPos("thresh1", "pars")

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

	find_colors_shapes(matched_shapes)


def get_shape_color(shape):
	return {
		"diamond": (255, 0, 0),
		"squiggle": (0, 255, 0),
		"oval": (0, 0, 255)
	}[shape]


def dist_squared(point0, point1):
	return (point1[0] - point0[0]) ** 2 + (point1[1] - point0[1])


def get_contour_scaled(contour, contour_mid, scale):
	contour_scaled = contour - contour_mid
	contour_scaled = contour_scaled * scale
	return np.int0(contour_scaled + contour_mid)


def normalized(vec):
	return vec / np.sqrt(vec[0] ** 2 + vec[1] ** 2)


def grow_contour(contour, pixels):
	shrunken_contour = np.zeros((0, 2))

	prev_point = contour[-2][0]
	next_point = contour[0][0]
	point = contour[-1][0]

	for i in range(1, len(contour) - 1):

		if not prev_point is next_point:
			dist = next_point - prev_point
			facing = normalized([dist[1], -dist[0]])
			shrunken_contour = np.append(shrunken_contour, [point + facing * pixels], 0)

		prev_point = point
		point = next_point
		next_point = contour[i][0]

	return np.int0(shrunken_contour)


def find_colors_shapes(shapes):
	global img
	global canvas

	unique_shapes = []

	for shape in shapes:

		if shape.child_contour is not None:
			continue

		is_unique = True

		for other in shapes:

			if shape is other:
				continue

			point = shape.contour[0][0]

			if cv.pointPolygonTest(other.contour, (point[0], point[1]), False) > 0:
				other.child_contour = shape.contour
				is_unique = False
				break

		if is_unique is True:
			unique_shapes.append(shape)

	for shape in unique_shapes:
		shape.parent_contour = grow_contour(shape.contour, shape.max_extent * 0.08)

		if shape.child_contour is None:
			shape.child_contour = grow_contour(shape.contour, -shape.max_extent * 0.08)

	# canvas = np.zeros(img.shape, np.uint8)

	for shape in unique_shapes:

		# mask = np.zeros(img.shape[:2], np.uint8)
		# cv.drawContours(mask, [shape.child_contour], -1, 255, -1)
		# mean = cv.mean(img, mask)
		# cv.drawContours(canvas, [shape.child_contour], -1, mean, -1)

		# hls_mean = gbr2hls(np.array(mean))

		mask = np.zeros(img.shape[:2], np.uint8)
		cv.drawContours(mask, [shape.contour], -1, 255, -1)
		cv.drawContours(mask, [shape.child_contour], -1, 0, -1)

		mean = cv.mean(img, mask)

		color_name = find_shape_color(gbr2hls(np.array(mean)))
		shape.color = color_name

		cv.drawContours(canvas, [shape.contour], -1, get_color_color(shape.color), 2)

		# mask = np.zeros(img.shape[:2], np.uint8)
		# cv.drawContours(mask, [shape.parent_contour], -1, 255, -1)
		# cv.drawContours(mask, [shape.contour], -1, 0, -1)
		# mean = cv.mean(img, mask)
		# cv.drawContours(canvas, [shape.parent_contour], -1, mean, 2)
		# cv.drawContours(canvas, [shape.contour], -1, 255, 1)

		hls_mean = gbr2hls(np.array(mean))


# print(int(hls_mean[1] / 255 * 360))

def find_shape_color(hls_color):
	hue = hls_color[0] * 360

	if hue >= 350 or hue <= 20:
		return "red"
	elif 240 <= hue <= 330:
		return "purple"
	elif 30 <= hue <= 160:
		return "green"
	else:
		return "other"


def get_color_color(color_name):
	return {
		"purple": (255, 0, 0),
		"green": (0, 255, 0),
		"red": (0, 0, 255),
		"other": 0
	}[color_name]


def gbr2hls(color):
	# print("rgb", (int(color[2]), int(color[1]), int(color[0])))
	color = color / 255
	h, l, s = colorsys.rgb_to_hls(color[2], color[1], color[0])
	# print("hsl", (int(h * 360), int(s * 100), int(l * 100)))

	# r, g, b = colorsys.hls_to_rgb(h, l, s)
	return np.array([h, l, s])


# cv.namedWindow("pars")
# cv.resizeWindow("pars", 640, 240)
# cv.createTrackbar("thresh1", "pars", 9, 30, update)
# cv.createTrackbar("thresh2", "pars", 20, 100, update)

kernelSize = 5
sigma = 3

img = cv.imread('./res/test_img.JPG')

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
