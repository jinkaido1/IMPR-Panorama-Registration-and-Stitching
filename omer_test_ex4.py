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
test_3_1_harris_corner_detector = True
test_3_1_sample_descriptor = True
test_3_1_find_features = True

"""
Methods
"""


def _test_harris_corner_detector():
	_start_test("3.1.1 - Harris Corner Detector")
	_notes("(1) Checks output size only!\n"
		   "(2) It plots an image, watch it carefully!\n"
		   "(3) Make sure all points are in the image.")
	im = read_image(IMG1, IMG_REP)
	arr = harris_corner_detector(im)
	plt.imshow(im, cmap='gray')
	plt.scatter(x=arr[:, 0], y=arr[:, 1], marker=".")
	plt.title("3.1.1 - Harris Corner Detector")
	plt.show()
	s = arr.shape
	assert len(s) == 2 and s[1] == 2
	_end_test("3.1.2 - Harris Corner Detector")


def _test_sample_descriptor():
	_start_test("3.1.2 - sample_descriptor")
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
	_end_test("3.1.2 - sample_descriptor")


def _test_find_features():
	_start_test("3.1.3 - find features")
	_notes("Checks output size only!")
	# Create 3 levels gaussian pyramid.
	im = read_image(IMG1, IMG_REP)
	pyr, _ = sol4_utils.build_gaussian_pyramid(im, 3, FILTER_SIZE)
	desc_rad = 3
	K = 1 + 2 * desc_rad
	returned_val = find_features(pyr)
	# pos.shape should be (N, 2)
	pos, descriptor = returned_val[0], returned_val[1]
	assert len(pos.shape) == 2 and pos.shape[1] == 2
	# descriptor.shape should be (N, K, K)
	assert len(descriptor.shape) == 3 and \
		   descriptor.shape[1] == descriptor.shape[2] == K
	_end_test("3.1.3 - find features")


"""
Callings.
"""
if test_3_1_harris_corner_detector:
	_test_harris_corner_detector()

if test_3_1_sample_descriptor:
	_test_sample_descriptor()

if test_3_1_find_features:
	_test_find_features()
