import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from scipy.signal import convolve2d
from scipy.ndimage.filters import convolve

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

# Minimum size of image after reducing.
MIN_SIZE = 16


"""
Given methods.
"""


def gaussian_kernel(kernel_size):
	conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
	conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
	kernel = np.array([1], dtype=np.float64)[:, None]
	for i in range(kernel_size - 1):
		kernel = convolve2d(kernel, conv_kernel, 'full')
	return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
	kernel = gaussian_kernel(kernel_size)
	blur_img = np.zeros_like(img)
	if len(img.shape) == 2:
		blur_img = convolve2d(img, kernel, 'same', 'symm')
	else:
		for i in range(3):
			blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
	return blur_img


"""
Helper methods from previous exercises.
"""


def read_image(filename, representation):
	"""
	:param filename: the file name of an image on the disk (could be grayscale or RGB_REP).
	:param representation: representation code, either 1 or 2 defining whether the output should be a gray scale
	image (1) or an RGB_REP image (2). If the input image is gray scale, we won't call it with representation = 2.

	Returns an image, represented by a matrix of type np.float64 with intensities
	(either grayscale or RGB_REP channel intensities) normalized to the range [0, 1].
	"""
	image = imread(filename).astype(np.float64)
	image /= MAX_COLOR
	if representation == GRAY_REP:
		image = rgb2gray(image)
	return image


def _build_binomial_coefficient_vector(filter_size):
	"""
	Builds the binomial coefficient vector.
	"""
	if filter_size == 1:
		return np.array([[1]])
	base = np.array([1, 1])
	vector = np.array([1, 1])
	while len(vector) < filter_size:
		vector = np.convolve(vector, base)
	sum_of_vector = sum(vector)
	return (vector / sum_of_vector).reshape(1, filter_size)


def _reduce(im, filter_vec):
	"""
	Reducing images of (X, X) to (X//2 , X//2).
	Takes the (odd1, odd2) for some two odd numbers from the image
	which has been gone convolution twice - both on the rows and on the columns.
	"""
	# Size of new image.
	im = convolve(im, filter_vec)
	im = convolve(im, filter_vec.T)
	return im[::2, ::2]


def build_gaussian_pyramid(im, max_levels, filter_size):
	"""
	:param im: a gray scale image with double values in [0; 1] (e.g. the output of ex1's read_image with
	the representation set to 1).
	:param max_levels: the maximal number of levels1 in the resulting pyramid.
	:param filter_size: the size of the Gaussian lter (an odd scalar that represents a squared lter) to be used
	in constructing the pyramid filter (e.g for filter_size = 3 you should get [0:25; 0:5; 0:25]).
	:return: (pyr, filter_vec)
			pyr - a standard python array (i.e. not
				numpy's array) with maximum length of max_levels, where each element of the array is a grayscale
				image.
			filter_vec - row vector of shape (1, filter_size) used for the pyramid construction.
	"""

	filter_vec = _build_binomial_coefficient_vector(filter_size)
	pyr = [im]

	curr_im = im
	while curr_im.shape[0] % 2 == 0 and \
			curr_im.shape[1] % 2 == 0 and \
			len(pyr) < max_levels and \
			curr_im.shape[0] >= 2 * MIN_SIZE and \
			curr_im.shape[1] >= 2 * MIN_SIZE:
		curr_im = _reduce(curr_im, filter_vec)
		pyr.append(curr_im)
	return pyr, filter_vec
