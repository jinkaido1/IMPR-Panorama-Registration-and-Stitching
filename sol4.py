"""
Exercise 4 - Image Processing 2019/2020
===
Author: Omer Liberman.
Start Date: Dec 07, 2019.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from imageio import imread
from skimage.color import rgb2gray

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