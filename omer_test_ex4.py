"""
Easy tester for ex4 - IMPR.
"""
from sol4 import *
from sol4_utils import *
import matplotlib.pyplot as plt

"""
Messages.
"""
START_TEST = "===== Starts test: {title} ====="
END_TEST = "----- End of test: {title} -----"
NOTES = "----- Notes: {note} -----"


def _start_test(input):
	print(START_TEST.format(title=input))


def _end_test(input):
	print(END_TEST.format(title=input) + "\n===== Check for errors! ======")


def _notes(input):
	print(NOTES.format(note=input))


"""
Constants.
"""
IMG_REP = 1

IMG1 = 'external/oxford1.jpg'
IMG2 = 'external/oxford2.jpg'
IMAGES = [IMG1, IMG2]

FILTER_SIZE = 5

"""
Controller.
"""
test_3_1_harris_corner_detector = False
test_3_1_sample_descriptor = True
test_3_1_find_features = False

"""
Methods
"""


def _test_harris_corner_detector():
	_start_test("3.1 - Harris Corner Detector")
	_notes("(1) Checks output size only!\n"
		   "(2) It plots an image, watch it carefully!\n"
		   "(3) Make sure all points are in the image.")
	im = read_image(IMG1, IMG_REP)
	arr = harris_corner_detector(im)
	plt.imshow(im, cmap='gray')
	plt.scatter(x=arr[:, 0], y=arr[:, 1], marker=".")
	plt.show()
	s = arr.shape
	assert len(s) == 2 and s[1] == 2
	_end_test("3.1 - Harris Corner Detector")


def _test_sample_descriptor():
	_start_test("3.1 - sample_descriptor")
	_notes("Checks output size only!")
	# Create 3 levels gaussian pyramid.
	im = read_image(IMG1, IMG_REP)
	pyr, filter = sol4_utils.build_gaussian_pyramid(im, 3, FILTER_SIZE)
	smallest_img = pyr[LEVELS - 1]
	desc_rad = 3
	K = 1 + 2 * desc_rad
	pos = harris_corner_detector(im)
	descriptors = sample_descriptor(smallest_img, pos, desc_rad)
	s = descriptors.shape
	assert len(s) == 3 and s[1] == s[2] == K
	_end_test("3.1 - sample_descriptor")


def _test_find_features():
	pass


"""
Callings.
"""
if test_3_1_harris_corner_detector:
	_test_harris_corner_detector()

if test_3_1_sample_descriptor:
	_test_sample_descriptor()

if test_3_1_find_features:
	_test_find_features()
