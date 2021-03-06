"""
Exercise 4 - Image Processing 2019/2020
Panorama Registration & Stitching
===
Author: Omer Liberman.
Start Date: Dec 24, 2019.
"""
import itertools
import os
import shutil
import numpy as np
import sol4_utils
import matplotlib.pyplot as plt

from imageio import imwrite
from scipy.ndimage import label, center_of_mass, map_coordinates
from scipy.ndimage.filters import maximum_filter
from scipy.signal import convolve2d
from scipy.ndimage.morphology import generate_binary_structure

"""
Constants.
"""
# The minimal value single pixel can have.
MIN_COLOR = 0.

# The maximal value single pixel can have.
MAX_COLOR = 255

# The number of legal values single pixel can have.
COLORS_NUM = 256

# Number of dimension in rgb image.
RGB_DIMS = 3

# Representation of gray-scale and rgb image.
GRAY_REP = 1
RGB_REP = 2

# Convolution vector for deriving.
DER_VEC = np.array([[1, 0, -1]])
DER_VEC_T = DER_VEC.T

# Blurring kernel size.
KERNEL_SIZE_BLUR = 3

K = 0.04                                                           # For calculating (det - k*trace).
LEVELS = 3                                                         # Max levels in gaussian
PATCH_SIZE = 7                                                     # Patch size.
ORIG_LEV_IN_PYR = 0                                                # Index of the original image in pyramid.
SMALLEST_LEV_IN_PYR = 2                                            # Index of the smallest image in pyramid.
LEVELS_DIFF = float(2 ** (ORIG_LEV_IN_PYR - SMALLEST_LEV_IN_PYR))  # Scalar to calculate new image index.
DESC_RAD = 3                                                       # The radius for the descriptor.
X, Y = 1, 0                                                        # for coordiantes.

"""
3 - Image Pair Registration.
"""


def harris_corner_detector(im):
	"""
	Detects harris corners.
	Make sure the returned coordinates are x major!!!
	:param im: A 2D array representing an image.
	:return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
	"""
	# Task 3.1.1
	Ix, Iy = convolve2d(im, DER_VEC, mode="same", boundary="symm"), \
			 convolve2d(im, DER_VEC_T, mode="same", boundary="symm")

	Ix_squared = Ix * Ix
	Iy_squared = Iy * Iy
	IxIy = Ix * Iy
	IyIx = Iy * Ix

	Ix_squared_blured = sol4_utils.blur_spatial(Ix_squared, KERNEL_SIZE_BLUR)
	Iy_squared_blured = sol4_utils.blur_spatial(Iy_squared, KERNEL_SIZE_BLUR)
	IxIy_blured = sol4_utils.blur_spatial(IxIy, KERNEL_SIZE_BLUR)
	IyIx_blured = sol4_utils.blur_spatial(IyIx, KERNEL_SIZE_BLUR)

	detM = (Ix_squared_blured * Iy_squared_blured) - (IxIy_blured * IyIx_blured)
	traceM = Ix_squared_blured + Iy_squared_blured
	R = detM - K * np.power(traceM, 2)
	non_max_sup = non_maximum_suppression(R)

	return np.argwhere(non_max_sup.transpose())


def sample_descriptor(im, pos, desc_rad):
	"""
	Samples descriptors at the given corners.
	:param im: A 2D array representing an image.
	:param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
	:param desc_rad: "Radius" of descriptors to compute.
	:return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
	"""
	# Task 3.1.2
	K = 1 + 2 * desc_rad
	N = len(pos)
	descriptors = np.zeros(shape=(N, K, K))
	x, y = np.meshgrid(np.arange(-desc_rad, desc_rad + 1), np.arange(-desc_rad, desc_rad + 1))
	for ind, p in enumerate(pos):
		posX, posY = p[0], p[1]
		patch = [(y + posY), (x + posX)]
		d_tilda = map_coordinates(im, patch, order=1, prefilter=False)
		Mu = d_tilda.mean()
		numerator = d_tilda - Mu
		denominator = np.linalg.norm((d_tilda - Mu))
		if denominator:
			descriptors[ind, :, :] = numerator / denominator
		else:
			descriptors[ind, :, :] = numerator
	return descriptors


def find_features(pyr):
	"""
	Detects and extracts feature points from a pyramid.
	:param pyr: Gaussian pyramid of a grayscale image having 3 levels.
	:return: A list containing:
				1) An array with shape (N,2) of [x,y] feature location per row found in the image.
				   These coordinates are provided at the pyramid level pyr[0].
				2) A feature descriptor array with shape (N,K,K)
	"""
	# Task 3.1.3
	orig_img, smallest_img = pyr[0], pyr[len(pyr) - 1]
	corners = spread_out_corners(orig_img, PATCH_SIZE, PATCH_SIZE, DESC_RAD * (1 / LEVELS_DIFF))
	return [corners, sample_descriptor(smallest_img, corners * LEVELS_DIFF, DESC_RAD)]


def match_features(desc1, desc2, min_score):
	"""
	Return indices of matching descriptors.
	:param desc1: A feature descriptor array with shape (N1,K,K).
	:param desc2: A feature descriptor array with shape (N2,K,K).
	:param min_score: Minimal match score.
	:return: A list containing:
				1) An array with shape (M,) and dtype int of matching indices in desc1.
				2) An array with shape (M,) and dtype int of matching indices in desc2.
	"""
	N1, N2 = desc1.shape[0], desc2.shape[0]
	K = desc1.shape[1]

	# Each descriptor turned line and multiple each matched line.
	desc1, desc2 = desc1.reshape(N1, K * K), desc2.reshape(N2, K * K)
	descriptors_mul = np.dot(desc1, desc2.T)

	# Find best 2 for each descriptor.
	x_axis, y_axis = np.sort(descriptors_mul, axis=0), np.sort(descriptors_mul, axis=1)
	best_x, best_y = x_axis[-2, :].reshape(1, N2), y_axis[:, -2].reshape(N1, 1)

	# Return 2 best matches.
	first_match, second_match = np.where((descriptors_mul >= best_x) &
										 (descriptors_mul >= best_y) &
										 (descriptors_mul > min_score))
	return [first_match, second_match]


def apply_homography(pos1, H12):
	"""
	Apply homography to inhomogenous points.
	:param pos1: An array with shape (N,2) of [x,y] point coordinates.
	:param H12: A 3x3 homography matrix.
	:return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
	"""
	pos2 = np.ones(shape=(pos1.shape[0], pos1.shape[1] + 1))
	pos2[:, :2] = pos1

	pos2 = pos2.T
	XYZdx = np.dot(H12, pos2).T

	XYdx, Zdx = XYZdx[:, :2], XYZdx[:, 2]

	XYdx[:, 0] /= Zdx
	XYdx[:, 1] /= Zdx
	return XYdx


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
	"""
	Recommendation: Use one of np.random.choice or np.random.permutation.
	Computes homography between two sets of points using RANSAC.
	:param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
	:param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
	:param num_iter: Number of RANSAC iterations to perform.
	:param inlier_tol: inlier tolerance threshold.
	:param translation_only: see estimate rigid transform
	:return: A list containing:
				1) A 3x3 normalized homography matrix.
				2) An Array with shape (S,) where S is the number of inliers,
					containing the indices in pos1/pos2 of the maximal set of inlier matches found.
	"""
	# Task 3.3.2
	P1, P2 = points1, points2
	N = points1.shape[0]
	Jin = []

	points_to_capture = 1 if translation_only else 2

	for it in range(num_iter):
		# Sample 2 sets of 2 points, anc calculate H12 base on the sets.
		idx = np.random.choice(N, points_to_capture, replace=False)
		P1j = points1[idx]
		P2j = points2[idx]
		H12 = estimate_rigid_transform(P1j, P2j, translation_only)

		# Calculate P2'.
		P2_tag = apply_homography(P1, H12)

		Ej = np.power(np.linalg.norm(P2_tag - P2, axis=1), 2)
		inliers = np.where(Ej < inlier_tol)[0]

		if len(inliers) > len(Jin):
			Jin = inliers

	P1Jin = np.array([P1[j] for j in Jin])
	P2Jin = np.array([P2[j] for j in Jin])
	homography_matrix = estimate_rigid_transform(P1Jin, P2Jin)
	return [homography_matrix, np.array(Jin)]


def display_matches(im1, im2, points1, points2, inliers):
	"""
	Dispalay matching points.
	:param im1: A grayscale image.
	:param im2: A grayscale image.
	:parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
	:param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
	:param inliers: An array with shape (S,) of inlier matches.
	"""
	# Task 3.3.3
	stacked_img = np.hstack([im1, im2])
	plt.imshow(stacked_img, cmap='gray')
	plt.axis('off')

	points2[:, 0] += im1.shape[1]

	plt.scatter(x=points1[:, 0], y=points1[:, 1], c='r', marker=".")
	plt.scatter(x=points2[:, 0], y=points2[:, 1], c='r', marker=".")

	# plot blue line.
	for idx in range(len(points1)):
		if idx not in inliers:
			p1 = points1[idx]
			p2 = points2[idx]
			plt.plot([p1[0], p2[0]], [p1[1], p2[1]], mfc='r', c='b', lw=.4, ms=1, marker='o')

	# plot yellow line.
	for idx in inliers:
		p1 = points1[idx]
		p2 = points2[idx]
		plt.plot([p1[0], p2[0]], [p1[1], p2[1]], mfc='r', c='y', lw=.5, ms=1, marker='o')

	plt.show()


def accumulate_homographies(H_succesive, m):
	"""
	Convert a list of succesive homographies to a
	list of homographies to a common reference frame.
	:param H_successive: A list of M-1 3x3 homography
	  matrices where H_successive[i] is a homography which transforms points
	  from coordinate system i to coordinate system i+1.
	:param m: Index of the coordinate system towards which we would like to
	  accumulate the given homographies.
	:return: A list of M 3x3 homography matrices,
	  where H2m[i] transforms points from coordinate system i to coordinate system m
	"""
	# Task 3.4
	H2m = []  # The returned list.
	M = len(H_succesive) + 1

	for i in range(M):
		Him = np.eye(3)
		if m > i:
			for j in range(i, min(m + 1, M)):
				Him = np.dot(Him, H_succesive[j])

		elif m < i:
			for j in range(m, i):
				Him = np.dot(Him, np.linalg.inv(H_succesive[j]))

		elif m == i:
			pass

		else:
			raise Exception('Undefined case.')

		Him /= Him[2, 2]
		H2m.append(Him)

	return H2m


def compute_bounding_box(homography, w, h):
	"""
	computes bounding box of warped image under homography, without actually warping the image
	:param homography: homography
	:param w: width of the image
	:param h: height of the image
	:return: 2x2 array, where the first row is [x,y] of the top left corner,
	 and the second row is the [x,y] of the bottom right corner
	"""
	# Task 4.1.1
	corners = np.array([[0, 0], [w, 0], [0, h], [w, h]])
	returned_coords = apply_homography(corners, homography)

	bottom_right = np.array([np.max(returned_coords[:, 0]), np.max(returned_coords[:, 1])])
	top_left = np.array([np.min(returned_coords[:, 0]), np.min(returned_coords[:, 1])])

	return np.array([top_left, bottom_right]).astype(np.int)


def warp_channel(image, homography):
	"""
	Warps a 2D image with a given homography.
	:param image: a 2D image.
	:param homography: homograhpy.
	:return: A 2d warped image.
	"""
	# Task 4.1.2
	h, w = image.shape
	corners = compute_bounding_box(homography, w, h)
	top_left, bottom_right = corners[0], corners[1]
	x, y = np.meshgrid(np.arange(top_left[0], bottom_right[0]),
					   np.arange(top_left[1], bottom_right[1]))
	coords = np.dstack([x, y])
	coords = coords.reshape(x.shape[0] * x.shape[1], 2)

	inv_homography = np.linalg.inv(homography)
	coords = apply_homography(coords, inv_homography).reshape(x.shape[0], x.shape[1], 2)

	new_img = map_coordinates(image, [coords[:, :, 1], coords[:, :, 0]], order=1, prefilter=False)
	return new_img


def warp_image(image, homography):
	"""
	Warps an RGB image with a given homography.
	:param image: an RGB image.
	:param homography: homograhpy.
	:return: A warped image.
	"""
	return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


"""
Given code.
"""


def filter_homographies_with_translation(homographies, minimum_right_translation):
	"""
	Filters rigid transformations encoded as homographies by the amount of translation from left to right.
	:param homographies: homograhpies to filter.
	:param minimum_right_translation: amount of translation below which the transformation is discarded.
	:return: filtered homographies..
	"""
	translation_over_thresh = [0]
	last = homographies[0][0, -1]
	for i in range(1, len(homographies)):
		if homographies[i][0, -1] - last > minimum_right_translation:
			translation_over_thresh.append(i)
			last = homographies[i][0, -1]
	return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
	"""
	Computes rigid transforming points1 towards points2, using least squares method.
	points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
	:param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
	:param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
	:param translation_only: whether to compute translation only. False (default) to compute rotation as well.
	:return: A 3x3 array with the computed homography.
	"""
	centroid1 = points1.mean(axis=0)
	centroid2 = points2.mean(axis=0)

	if translation_only:
		rotation = np.eye(2)
		translation = centroid2 - centroid1

	else:
		centered_points1 = points1 - centroid1
		centered_points2 = points2 - centroid2

		sigma = centered_points2.T @ centered_points1
		U, _, Vt = np.linalg.svd(sigma)

		rotation = U @ Vt
		translation = -rotation @ centroid1 + centroid2

	H = np.eye(3)
	H[:2, :2] = rotation
	H[:2, 2] = translation
	return H


def non_maximum_suppression(image):
	"""
	Finds local maximas of an image.
	:param image: A 2D array representing an image.
	:return: A boolean array with the same shape as the input image, where True indicates local maximum.
	"""
	# Find local maximas.
	neighborhood = generate_binary_structure(2, 2)
	local_max = maximum_filter(image, footprint=neighborhood) == image
	local_max[image < (image.max() * 0.1)] = False

	# Erode areas to single points.
	lbs, num = label(local_max)
	centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
	centers = np.stack(centers).round().astype(np.int)
	ret = np.zeros_like(image, dtype=np.bool)
	ret[centers[:, 0], centers[:, 1]] = True

	return ret


def spread_out_corners(im, m, n, radius):
	"""
	Splits the image im to m by n rectangles and uses harris_corner_detector on each.
	:param im: A 2D array representing an image.
	:param m: Vertical number of rectangles.
	:param n: Horizontal number of rectangles.
	:param radius: Minimal distance of corner points from the boundary of the image.
	:return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
	"""
	corners = [np.empty((0, 2), dtype=np.int)]
	x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
	y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
	for i in range(n):
		for j in range(m):
			# Use Harris detector on every sub image.
			sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
			sub_corners = harris_corner_detector(sub_im)
			sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
			corners.append(sub_corners)
	corners = np.vstack(corners)
	legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
			 (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
	ret = corners[legit, :]
	return ret


class PanoramicVideoGenerator:
	"""
	Generates panorama from a set of images.
	"""

	def __init__(self, data_dir, file_prefix, num_images):
		"""
		The naming convention for a sequence of images is file_prefixN.jpg,
		where N is a running number 001, 002, 003...
		:param data_dir: path to input images.
		:param file_prefix: see above.
		:param num_images: number of images to produce the panoramas with.
		"""
		self.file_prefix = file_prefix
		self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
		self.files = list(filter(os.path.exists, self.files))
		self.panoramas = None
		self.homographies = None
		print('found %d images' % len(self.files))

	def align_images(self, translation_only=False):
		"""
		compute homographies between all images to a common coordinate system
		:param translation_only: see estimte_rigid_transform
		"""
		# Extract feature point locations and descriptors.
		points_and_descriptors = []
		for file in self.files:
			image = sol4_utils.read_image(file, 1)
			self.h, self.w = image.shape
			pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
			points_and_descriptors.append(find_features(pyramid))

		# Compute homographies between successive pairs of images.
		Hs = []
		for i in range(len(points_and_descriptors) - 1):
			points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
			desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

			# Find matching feature points.
			ind1, ind2 = match_features(desc1, desc2, .7)
			points1, points2 = points1[ind1, :], points2[ind2, :]

			# Compute homography using RANSAC.
			H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

			# Uncomment for debugging: display inliers and outliers among matching points.
			# In the submitted code this function should be commented out!
			# display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

			Hs.append(H12)

		# Compute composite homographies from the central coordinate system.
		accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
		self.homographies = np.stack(accumulated_homographies)
		self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
		self.homographies = self.homographies[self.frames_for_panoramas]

	def generate_panoramic_images(self, number_of_panoramas):
		"""
		combine slices from input images to panoramas.
		:param number_of_panoramas: how many different slices to take from each input image
		"""
		assert self.homographies is not None

		# compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
		self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
		for i in range(self.frames_for_panoramas.size):
			self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

		# change our reference coordinate system to the panoramas
		# all panoramas share the same coordinate system
		global_offset = np.min(self.bounding_boxes, axis=(0, 1))
		self.bounding_boxes -= global_offset

		slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
		warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
		# every slice is a different panorama, it indicates the slices of the input images from which the panorama
		# will be concatenated
		for i in range(slice_centers.size):
			slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
			# homography warps the slice center to the coordinate system of the middle image
			warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
			# we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
			warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

		panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

		# boundary between input images in the panorama
		x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
		x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
									  x_strip_boundary,
									  np.ones((number_of_panoramas, 1)) * panorama_size[0]])
		x_strip_boundary = x_strip_boundary.round().astype(np.int)

		self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
		for i, frame_index in enumerate(self.frames_for_panoramas):
			# warp every input image once, and populate all panoramas
			image = sol4_utils.read_image(self.files[frame_index], 2)
			warped_image = warp_image(image, self.homographies[i])
			x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
			y_bottom = y_offset + warped_image.shape[0]

			for panorama_index in range(number_of_panoramas):
				# take strip of warped image and paste to current panorama
				boundaries = x_strip_boundary[panorama_index, i:i + 2]
				image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
				x_end = boundaries[0] + image_strip.shape[1]
				self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

		# crop out areas not recorded from enough angles
		# assert will fail if there is overlap in field of view between the left most image and the right most image
		crop_left = int(self.bounding_boxes[0][1, 0])
		crop_right = int(self.bounding_boxes[-1][0, 0])
		assert crop_left < crop_right
		print(crop_left, crop_right)
		self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

	def save_panoramas_to_video(self):
		assert self.panoramas is not None
		out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
		try:
			shutil.rmtree(out_folder)
		except:
			print('could not remove folder')
			pass
		os.makedirs(out_folder)
		# save individual panorama images to 'tmp_folder_for_panoramic_frames'
		for i, panorama in enumerate(self.panoramas):
			imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
		if os.path.exists('%s.mp4' % self.file_prefix):
			os.remove('%s.mp4' % self.file_prefix)
		# write output video to current folder
		os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
				  (out_folder, self.file_prefix))

	def show_panorama(self, panorama_index, figsize=(20, 20)):
		assert self.panoramas is not None
		plt.figure(figsize=figsize)
		plt.imshow(self.panoramas[panorama_index].clip(0, 1))
		plt.show()
