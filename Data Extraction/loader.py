"""
load downloaded files for use in models

Need to load dataset pairs

"""

import numpy as np
import tensorflow as tf
import pathlib
import functools
import util
import rasterio as rio
import tensorflow_io as tfio
AUTOTUNE = tf.data.experimental.AUTOTUNE
"""
NOTES:

Modis and landsat sizes differ:
Landsat has dimensions: 7851x7971
MODIS has dimensions: 2590x2662

"""
class landsat_modis_loader(object):
    def __init__(self, base_dir):

        self._base_dir = pathlib.Path(base_dir)
        self.pairs = util.get_landsat_modis_pairs(self._base_dir, transform=True)
        self.num_pairs = len(self.pairs)
        # we take first image samples to get height and width
        L_sample = rio.open(self.pairs[0][0])
        M_sample = rio.open(self.pairs[0][1])

        self._L_width, self._L_height = L_sample.width, L_sample.height
        self._M_width, self._M_height = M_sample.width, M_sample.height
        self.l_bands = (3,2,1)
        self.m_bands = (1, 4, 3)

        self._current_ds = {}






