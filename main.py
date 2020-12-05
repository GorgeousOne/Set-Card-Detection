import numpy as np
import cv2 as cv
import colorsys


class SetShape:

	def __init__(self, contour, bbox, exact_rect, shape, color=None, shading=None, child_contour=None,
				 parent_contour=None):
		self.contour = contour
		self.bbox = bbox
		self.exact_rect = exact_rect
		self.shape = shape
		self.color = color
		self.shading = shading
		self.child_contour = child_contour
		self.parent_contour = parent_contour
		self.max_extent = max(exact_rect[1][0], exact_rect[1][0])


def get_blurred_gray(img):
	kernel_size = 5

	img_blur = cv.GaussianBlur(img, (kernel_size, kernel_size), cv.BORDER_DEFAULT)
	return cv.cvtColor(img_blur, cv.COLOR_BGR2GRAY)


def find_contours(img_gray):
	thresh = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, 2)
	contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	return contours


def find_possible_shapes(contours, img_min_extent):
	min_contour_points = img_min_extent * 0.04
	min_bounds_size = img_min_extent * 0.03
	min_exact_bound_size = img_min_extent * 0.025
	max_exact_bound_size = img_min_extent * 0.19
	min_bounds_occupation = 0.55
	max_bounds_occupation = 0.90

	matched_shapes = []

	for contour in contours:

		# ignore short contours
		if len(contour) < min_contour_points:
			# cv.drawContours(canvas, [contour], -1, (0, 0, 0), 1)
			continue

		x, y, w, h = cv.boundingRect(contour)

		# ignore small dots
		if w < min_bounds_size and h < min_bounds_size:
			# cv.drawContours(canvas, [contour], -1, (255, 0, 0), 1)
			continue

		rect = cv.minAreaRect(contour)

		width = rect[1][0]
		height = rect[1][1]

		# ignore quiet thin things
		if width < min_exact_bound_size or height < min_exact_bound_size:
			# cv.drawContours(canvas, [contour], -1, (255, 128, 128), 1)
			continue

		if width > max_exact_bound_size and height > max_exact_bound_size:
			# cv.drawContours(canvas, [contour], -1, (0, 0, 255), 1)
			continue

		if not ratio_fits(rect):
			cv.drawContours(canvas, [contour], -1, (0, 0, 0), 1)
			continue

		area = cv.contourArea(contour)
		bounds_occupation = area / (width * height)

		if bounds_occupation < min_bounds_occupation:
			# cv.drawContours(canvas, [contour], -1, (255, 255, 255), 1)
			continue

		if bounds_occupation > max_bounds_occupation:
			# cv.drawContours(canvas, [contour], -1, (255, 255, 255), 1)
			continue

		set_shape = SetShape(contour, [x, y, w, h], rect, find_shape_type(bounds_occupation))
		matched_shapes.append(set_shape)

	return matched_shapes


# def get_aspect_ratio(rect):
# 	return rect[1][1] / rect[1][0]


def ratio_fits(rect):
	min_ratio = 1.3
	max_ratio = 3.2

	ratio = get_greater_aspect_ratio(rect)
	return min_ratio < ratio < max_ratio


def get_greater_aspect_ratio(rect):
	width = rect[1][0]
	height = rect[1][1]
	return (width / height) if width > height else (height / width)


def find_shape_type(bounds_occupation):
	return "diamond" if bounds_occupation < 0.65 else \
		"squiggle" if bounds_occupation < 0.81 else \
			"oval"


# def get_shape_color(shape):
# 	return {
# 		"diamond": (255, 0, 0),
# 		"squiggle": (0, 255, 0),
# 		"oval": (0, 0, 255)
# 	}[shape]


# def dist_squared(point0, point1):
# 	return (point1[0] - point0[0]) ** 2 + (point1[1] - point0[1])


def get_contour_scaled(contour, contour_mid, scale):
	contour_scaled = contour - contour_mid
	contour_scaled = contour_scaled * scale
	return np.int0(contour_scaled + contour_mid)


def normalized(vec):
	return np.array(vec) / np.sqrt(vec[0] ** 2 + vec[1] ** 2)


def grow_contour(contour, pixels):
	new_contour = np.zeros((0, 2))
	prev_point = contour[-2][0]
	next_point = contour[0][0]
	point = contour[-1][0]

	for i in range(1, len(contour) - 1):

		dist = next_point - prev_point

		if np.any(dist):
			facing = normalized([dist[1], -dist[0]])
			new_contour = np.append(new_contour, [point + facing * pixels], 0)

		prev_point = point
		point = next_point
		next_point = contour[i][0]

	return np.int0(new_contour)


def find_actual_shapes(shapes):
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

	return unique_shapes


def analyse_shapes_colors(shapes, img_colored):
	global canvas

	for shape in shapes:

		mask = np.zeros(img_colored.shape[:2], np.uint8)
		cv.drawContours(mask, [shape.contour], -1, 255, -1)
		cv.drawContours(mask, [shape.child_contour], -1, 0, -1)
		mean_contour = cv.mean(img_colored, mask)

		shape.color = find_shape_color(gbr2hls(mean_contour))
		cv.drawContours(canvas, [shape.contour], -1, get_color_for_color_name(shape.color), 2)

		mask = np.zeros(img_colored.shape[:2], np.uint8)
		cv.drawContours(mask, [shape.child_contour], -1, 255, -1)
		mean_inside = cv.mean(img_colored, mask)
		# cv.drawContours(canvas, [shape.child_contour], -1, mean_contour, -1)

		mask = np.zeros(img_colored.shape[:2], np.uint8)
		cv.drawContours(mask, [shape.parent_contour], -1, 255, -1)
		cv.drawContours(mask, [shape.contour], -1, 0, -1)
		mean_outside = cv.mean(img_colored, mask)

		shape.shading = find_shading(gbr2hls(mean_inside), gbr2hls(mean_outside))

		if shape.shading == "solid":
			cv.drawContours(canvas, [shape.child_contour], -1, 0, 2)
		elif shape.shading == "open":
			cv.drawContours(canvas, [shape.child_contour], -1, (255, 255, 255), 2)


def gbr2hls(color):
	# print("rgb", (int(color[2]), int(color[1]), int(color[0])))
	color = np.array(color) / 255
	h, l, s = colorsys.rgb_to_hls(color[2], color[1], color[0])
	return np.array([h, l, s])


def find_shape_color(hls_color):
	hue = hls_color[0] * 360
	if hue >= 350 or hue <= 20:
		return "red"
	elif 240 <= hue <= 330:
		return "purple"
	elif 30 <= hue <= 160:
		return "green"
	else:
		print(int(hue))
		return "other"


def find_shading(hsl_color_inside, hsl_color_outside):
	light_inside = hsl_color_inside[1] * 100
	light_outside = hsl_color_outside[1] * 100
	fall_off = light_outside - light_inside

	if fall_off < 4:
		return "open"
	elif fall_off < 15:
		return "striped"
	else:
		return "solid"


def get_color_for_color_name(color_name):
	return {
		"purple": (255, 1280, 128),
		"green": (0, 255, 0),
		"red": (0, 0, 255),
		"other": 0
	}[color_name]


def update(val):
	global imgGray
	global imgOrig
	global canvas

	canvas = imgOrig.copy()
	img_min_extent = min(imgOrig.shape[0], imgOrig.shape[1])

	contours = find_contours(imgGray)
	possible_shapes = find_possible_shapes(contours, img_min_extent)

	actual_shapes = find_actual_shapes(possible_shapes)
	analyse_shapes_colors(actual_shapes, imgOrig)


def get_img_resized(img):
	img_height = img.shape[0]
	img_width = img.shape[1]
	max_factor = max(img_width, img_height) // 1000  # or whatever

	if max_factor > 1:
		return cv.resize(img, (img_width // max_factor, img_height // max_factor))
	else:
		return img


cv.namedWindow("pars")
cv.resizeWindow("pars", 640, 240)
cv.createTrackbar("thresh1", "pars", 7, 30, update)
cv.createTrackbar("thresh2", "pars", 20, 100, update)

imgOrig = cv.imread('./res/test21.JPG')

if imgOrig is None:
	raise Exception("image not found")

imgOrig = get_img_resized(imgOrig)
imgGray = get_blurred_gray(imgOrig)
canvas = imgOrig.copy()
update(None)

while True:
	cv.imshow('image', canvas)
	cv.waitKey(1000)

	if cv.getWindowProperty('image', 0) < 0:
		exit()
