"""
Read tiffs as stacked rasters

investigate scene meta data
"""
import os


from glob import glob  # File manipulation
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import rasterio as rio
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

import util

def plot_cloud_cover(data):
    fig1, ax1 = plt.subplots()
    ax1.set_title('Landsat Cloud Cover Indexes %')
    ax1.boxplot(data)
    plt.show()


def plot_spatial_footprints(data):
    """
    plot spatial footprints for observations
    :param data: list of polygon tuples
    :return: None
    """
    data = iter(data)
    fig2, ax = plt.subplots(3, 3)
    for i in range(3):
        for n in range(3):
            pair = next(data)
            l_x, l_y = pair[0].exterior.xy
            m_x, m_y = pair[1].exterior.xy

            ax[i, n].plot(l_x, l_y,  alpha=0.4, color='b',
                          linewidth=3, solid_capstyle='round', zorder=2)
            ax[i, n].plot( m_x, m_y, alpha=0.4, color='r',
                           linewidth = 3, solid_capstyle = 'round', zorder = 2)

    fig2.suptitle('coordinate overlap display')
    plt.legend()
    plt.show()









"""
EXAMPLE OF RASTERIO USAGE
os.chdir(os.path.join(et.io.HOME, 'earth-analytics'))
# Get list of all pre-cropped data and sort the data

path = os.path.join("data", "cold-springs-fire", "landsat_collect",
                    "LC080340322016072301T1-SC20180214145802", "crop")

all_landsat_post_bands = glob(path + "/*band*.tif")
print(all_landsat_post_bands)
all_landsat_post_bands.sort()

# Create an output array of all the landsat data stacked
landsat_post_fire_path = os.path.join("data", "cold-springs-fire",
                                      "outputs", "landsat_post_fire.tif")

# This will create a new stacked raster with all bands
land_stack, land_meta = es.stack(all_landsat_post_bands,
                                 landsat_post_fire_path)
# read new stack
with rio.open(landsat_post_fire_path) as src:
    landsat_post_fire = src.read()

ep.plot_rgb(landsat_post_fire,
            rgb=[3, 2, 1],
            title="RGB Composite Image\n Post Fire Landsat Data")
plt.show()
"""