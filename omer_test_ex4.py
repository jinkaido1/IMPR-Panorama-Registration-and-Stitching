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
MIN_SCORE = 0.8
num_iter, inlier_tol = 50, 2

"""
Controller.
"""
test_3_1_harris_corner_detector =    False
test_3_1_sample_descriptor =         False
test_3_1_find_features =             False
test_3_2_match_features =            False
test_3_3_apply_homography =          False
test_3_3_ransac_homography =         False
test_3_3_display_matches =           False
test_3_4_accumulate_homographies =   False
test_4_1_compute_bounding_box =      True


"""
Methods
"""


def _test_harris_corner_detector():
	_start_test("3.1.1 - Harris Corner Detector")
	_notes("\n"
		   "(1) Checks output size only!\n"
		   "(2) It plots an image, watch it carefully!\n"
		   "(3) Make sure all points are in the image.")
	im1 = read_image(IMG1, IMG_REP)
	arr1 = harris_corner_detector(im1)
	plt.imshow(im1, cmap='gray')
	plt.scatter(x=arr1[:, 0], y=arr1[:, 1], marker=".")
	plt.title("3.1.1 - Harris Corner Detector")
	plt.show()
	s1 = arr1.shape

	im2 = read_image(IMG2, IMG_REP)
	arr2 = harris_corner_detector(im2)
	plt.imshow(im2, cmap='gray')
	plt.scatter(x=arr2[:, 0], y=arr2[:, 1], marker=".")
	plt.title("3.1.1 - Harris Corner Detector")
	plt.show()
	s2 = arr2.shape
	print("Found: {} corners in the first image, {} corners in the second image.".format(len(arr1), len(arr2)))
	assert len(s1) == 2 and s1[1] == 2
	assert len(s2) == 2 and s2[1] == 2
	_end_test("3.1.2 - Harris Corner Detector")


def _test_sample_descriptor():
	_start_test("3.1.2 - sample_descriptor")
	_notes("\n"
		   "(1) Checks output size only!\n"
		   "(2) It is not suppose to make the method work, just makes sure it runs.\n"
		   "(3) Might have zero division error."
		   "\n")
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
	im1 = read_image(IMG1, IMG_REP)
	pyr1, _ = sol4_utils.build_gaussian_pyramid(im1, 3, FILTER_SIZE)
	desc_rad = 3
	K = 1 + 2 * desc_rad
	returned_val1 = find_features(pyr1)
	# pos.shape should be (N, 2)
	pos1, descriptor1 = returned_val1[0], returned_val1[1]
	assert len(pos1.shape) == 2 and pos1.shape[1] == 2
	# descriptor.shape should be (N, K, K)
	assert len(descriptor1.shape) == 3 and descriptor1.shape[1] == descriptor1.shape[2] == K

	im2 = read_image(IMG2, IMG_REP)
	pyr2, _ = sol4_utils.build_gaussian_pyramid(im2, 3, FILTER_SIZE)
	desc_rad = 3
	K = 1 + 2 * desc_rad
	returned_val2 = find_features(pyr2)
	# pos.shape should be (N, 2)
	pos2, descriptor2 = returned_val2[0], returned_val2[1]
	assert len(pos2.shape) == 2 and pos2.shape[1] == 2
	# descriptor.shape should be (N, K, K)
	assert len(descriptor2.shape) == 3 and descriptor2.shape[1] == descriptor2.shape[2] == K

	print("Number of features found in img1: {}\n"
		  "Number of features found in img2: {}".format(len(pos1), len(pos2)))
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
	pos1, desc1 = returned_val1[0], returned_val1[1]

	im2 = read_image(IMG2, IMG_REP)
	pyr2, _ = sol4_utils.build_gaussian_pyramid(im2, 3, FILTER_SIZE)
	returned_val2 = find_features(pyr2)
	pos2, desc2 = returned_val2[0], returned_val2[1]

	# Run method:
	returned_val_func = match_features(desc1, desc2, MIN_SCORE)
	matching_desc1, matching_desc2 = returned_val_func[0], returned_val_func[1]

	# Plot number one:
	points1 = [pos1[i] for i in matching_desc1]
	xs1 = [a[0] for a in points1]
	ys1 = [a[1] for a in points1]
	plt.imshow(im1, cmap='gray')
	plt.scatter(x=xs1, y=ys1, marker=".")
	plt.title("3.2 - Match Features - 1st img")
	plt.show()

	# Plot number two:
	points2 = [pos2[j] for j in matching_desc2]
	xs2 = [a[0] for a in points2]
	ys2 = [a[1] for a in points2]
	plt.imshow(im2, cmap='gray')
	plt.scatter(x=xs2, y=ys2, marker=".")
	plt.title("3.2 - Match Features - 2nd img")
	plt.show()

	print("-- Num of matched points found: {}".format(len(matching_desc1)))
	# print("-- Matching 1 ---\n", matching_desc1, '\n')
	# print("-- Matching 2 ---\n", matching_desc2, '\n')

	assert len(matching_desc1.shape) == len(matching_desc2.shape) == 1
	assert matching_desc1.shape == matching_desc2.shape

	_end_test("3.2 - Match Features")


def _test_apply_homography():
	_start_test("3.3.1 - apply_homography")
	_notes("Checks output size only!")
	im1 = read_image(IMG1, IMG_REP)
	pyr1, _ = sol4_utils.build_gaussian_pyramid(im1, 3, FILTER_SIZE)
	returned_val1 = find_features(pyr1)
	pos1, _ = returned_val1[0], returned_val1[1]
	H12 = np.array([[1, 0, 0],
					[0, 1, 0],
					[0, 0, 1]])
	result = apply_homography(pos1, H12)
	assert result.shape == pos1.shape
	_end_test("3.3.1 - apply_homography")


def _test_rasnac_homography():
	_start_test("3.3.2 - rasnac_homography")
	_notes("Checks output size only!")
	im1 = read_image(IMG1, IMG_REP)
	pyr1, _ = sol4_utils.build_gaussian_pyramid(im1, 3, FILTER_SIZE)
	returned_val1 = find_features(pyr1)
	pos1, desc1 = returned_val1[0], returned_val1[1]

	im2 = read_image(IMG2, IMG_REP)
	pyr2, _ = sol4_utils.build_gaussian_pyramid(im2, 3, FILTER_SIZE)
	returned_val2 = find_features(pyr2)
	pos2, desc2 = returned_val2[0], returned_val2[1]

	# Run method:
	returned_val_func = match_features(desc1, desc2, MIN_SCORE)
	matching_desc1, matching_desc2 = returned_val_func[0], returned_val_func[1]

	points1 = np.array([pos1[i] for i in matching_desc1])
	points2 = np.array([pos2[j] for j in matching_desc2])

	result = ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False)
	homography_matrix, inliers_indexes = result[0], result[1]

	print("Num of inliers found: ", len(inliers_indexes))
	# Check matrix sizes.
	assert homography_matrix.shape[0] == homography_matrix.shape[1] == 3
	# Check inliers array shape.
	assert len(inliers_indexes.shape) == 1
	_end_test("3.3.1 - rasnac_homography")


def _test_3_3_display_matches():
	_start_test("3.3.3 - display_matches")
	_notes("Check the output images!")
	im1 = read_image(IMG1, IMG_REP)
	pyr1, _ = sol4_utils.build_gaussian_pyramid(im1, 3, FILTER_SIZE)
	returned_val1 = find_features(pyr1)
	pos1, desc1 = returned_val1[0], returned_val1[1]

	im2 = read_image(IMG2, IMG_REP)
	pyr2, _ = sol4_utils.build_gaussian_pyramid(im2, 3, FILTER_SIZE)
	returned_val2 = find_features(pyr2)
	pos2, desc2 = returned_val2[0], returned_val2[1]

	returned_val_func = match_features(desc1, desc2, MIN_SCORE)
	matching_desc1, matching_desc2 = returned_val_func[0], returned_val_func[1]
	points1 = np.array([pos1[i] for i in matching_desc1])
	points2 = np.array([pos2[j] for j in matching_desc2])

	result = ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False)
	homography_matrix, inliers = result[0], result[1]
	display_matches(im1, im2, points1, points2, inliers)
	_end_test("3.3.3 - display_matches")


def _test_3_4_accumulate_homographies():
	_start_test("3.4 - accumulate_homographies")
	_notes("\n(1) Checks sizes.\n(2) Check value (very simple case).\n")

	# Creating some random 3*3 matrices.
	H_succesive, m, M = [], 20, 20
	for i in range(20):
		if i < 19:
			new_mat = np.array([[1, 0, 0],
								[0, 1, 0],
								[0, 0, 1]])
		else:
			new_mat = np.array([[i, 0, 0],
								[0, i, 0],
								[0, 0, i]])
		H_succesive.append(new_mat)

	H2m_1 = accumulate_homographies(H_succesive, m)
	l = len(H2m_1)
	s = H2m_1[0].shape
	last_mat = H2m_1[M - 1]

	# Shape check
	assert len(s) + 1 == 3
	assert s[0] == s[1] == 3
	assert l == M
	assert np.array_equal(last_mat, np.array([[1., 0, 0],
											 [0, 1., 0],
											 [0, 0, 1.]]))

	_end_test("3.4 - accumulate_homographies")


def _test_4_1_compute_bounding_box():
	_start_test("4.1 - compute_bounding_box")
	_notes("\n(1) Checks sizes.\n")

	homography = np.array([[2., 0, 1.],
						 	[5., 2., 0],
							[0, 0, 1.]])
	h, w = 10, 20
	returned = compute_bounding_box(homography, h, w)

	assert returned.shape == (2, 2)
	assert returned.dtype == np.int
	_end_test("4.1 - compute_bounding_box")

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

if test_3_3_apply_homography:
	_test_apply_homography()

if test_3_3_ransac_homography:
	_test_rasnac_homography()

if test_3_3_display_matches:
	_test_3_3_display_matches()

if test_3_4_accumulate_homographies:
	_test_3_4_accumulate_homographies()

if test_4_1_compute_bounding_box:
	_test_4_1_compute_bounding_box()
