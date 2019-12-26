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
	print(END_TEST.format(title=input) + "\n===== Check for errors! ======\n")


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

MIN_SCORE = 0.5

"""
Controller.
"""
test_3_1_harris_corner_detector = False
test_3_1_sample_descriptor = False
test_3_1_find_features = False
test_3_2_match_features = True


"""
Methods
"""


def _test_harris_corner_detector():
	_start_test("3.1.1 - Harris Corner Detector")
	_notes("\n"
		   "(1) Checks output size only!\n"
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


def _test_match_features():
	_start_test("3.2.1 - match_features")
	_notes("\n"
		   "(1) Checks output size only!\n"
		   "(2) It plots 2 images! watch them carefully!\n"
		   "(3) Make sure all points are in the images.")
	im1 = read_image(IMG1, IMG_REP)
	pyr1, _ = sol4_utils.build_gaussian_pyramid(im1, 3, FILTER_SIZE)
	returned_val1 = find_features(pyr1)
	_, desc1 = returned_val1[0], returned_val1[1]

	im2 = read_image(IMG2, IMG_REP)
	pyr2, _ = sol4_utils.build_gaussian_pyramid(im2, 3, FILTER_SIZE)
	returned_val2 = find_features(pyr2)
	_, desc2 = returned_val2[0], returned_val2[1]

	# Run method:
	returned_val_func = match_features(desc1, desc2, MIN_SCORE)
	matching_desc1, matching_desc2 = returned_val_func[0], returned_val_func[1]

	print("-- Matching 1 ---\n", matching_desc1, '\n')
	print("-- Matching 2 ---\n", matching_desc2, '\n')

	assert len(matching_desc1.shape) == len(matching_desc2.shape) == 2
	assert matching_desc1.shape == matching_desc2.shape


"""
Callings.
"""
if test_3_1_harris_corner_detector:
	_test_harris_corner_detector()

if test_3_1_sample_descriptor:
	_test_sample_descriptor()

if test_3_1_find_features:
	_test_find_features()

if test_3_2_match_features:
	_test_match_features()