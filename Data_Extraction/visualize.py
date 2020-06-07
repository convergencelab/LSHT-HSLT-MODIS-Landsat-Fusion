"""
Read tiffs as stacked rasters

investigate scene meta data
"""
import os
from osgeo import gdal
from glob import glob  # File manipulation
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show, show_hist
import numpy as np
# import geopandas as gpd
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
def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min) / (array_max - array_min))

def plot_rgb(fpath, rgb):

    raster = rio.open(fpath)
    red = raster.read(rgb[0])
    green = raster.read(rgb[1])
    blue = raster.read(rgb[2])

    # Normalize the bands
    redn = normalize(red)
    greenn = normalize(green)
    bluen = normalize(blue)

    # Create RGB natural color composite
    rgb = np.dstack((redn, greenn, bluen))

    # Let's see how our color composite looks like
    return plt.imshow(rgb)


def plot_raster_pair():
    """
    view landsat modis pair
    :param im1: str path
    :param im2:  str path
    :return:
    """

    dir = os.environ['LS_MD_PAIRS']
    pairs = util.get_landsat_modis_pairs(dir)
    fig3, ax3 = plt.subplots(2, 2)

    with rio.open(glob(pairs[0][0] + "\*")[0]) as l_src:
        img1 = l_src.read()

        ax3[0, 0] = ep.plot_rgb(img1,
                    rgb=[3, 2, 1],
                    title="Landsat RGB Image\n Linear Stretch Applied",
                    stretch=True,
                    str_clip=4)

        ax3[0, 1]  = ep.plot_rgb(img1,
                    rgb=[3, 2, 1],
                    title="Landsat RGB Image",
                    stretch=False)


    with rio.open(glob(pairs[0][0] + "\*")[1]) as m_src:
        img2 = m_src.read()

        ax3[1, 0]  = ep.plot_rgb(img2,
                    rgb=[3, 2, 1],
                    title="MODIS RGB Image\n Linear Stretch Applied",
                    stretch=True,
                    str_clip=4)

        ax3[1, 1]  = ep.plot_rgb(img2,
                          rgb=[3, 2, 1],
                          title="MODIS RGB Image",
                          stretch=False)

        plt.show()

# plot_raster_pair()

def plot_ep_plot(imgs=None, stretch=None):
    """
    helper for visualization
    :param imgs: tuple of paths to image
    :param stretch: linear stretch bool
    :return: None
    """

    if not imgs:
        dir = os.environ['LS_MD_PAIRS']
        pairs = util.get_landsat_modis_pairs(dir, transform=True, both_modis=True)
        imgs = pairs[0]

    fig4, ax4 = plt.subplots(1, 2)
    # plot landsat
    with rio.open(imgs[0]) as l_src:
        img1 = l_src.read()
        ep.plot_rgb(
                img1,
                rgb=(3, 2, 1),
                figsize=(10, 10),
                str_clip=2,
                ax=ax4[0],
                extent=None,
                title="Landsat True Colour",
                stretch=stretch,
        )

    # Plot MODIS
    with rio.open(imgs[1]) as m_src:
        img2 = m_src.read()
        ep.plot_rgb(
            img2,
            rgb=(0, 3, 2),
            figsize=(10, 10),
            str_clip=2,
            ax=ax4[1],
            extent=None,
            title="MODIS True Colour",
            stretch=stretch,
        )

def show_affine_transform(imgs=False, stretch=True):
    """
     helper for visualization
     :param imgs: tuple of paths to image
     :param stretch: linear stretch bool
     :return: None
    """
    if not imgs:
        dir = os.environ['LS_MD_PAIRS']
        pairs = util.get_landsat_modis_pairs(dir, transform=True, both_modis=True)
        imgs = pairs[0]

    fig5, ax5 = plt.subplots(1, 2)
    # Plot MODIS untransformed
    with rio.open(pairs[0][0]) as m_src:
        img2 = m_src.read()
        ep.plot_rgb(
            img2,
            rgb=(0, 3, 2),
            figsize=(10, 10),
            str_clip=2,
            ax=ax5[0],
            extent=None,
            title="MODIS Untransformed",
            stretch=stretch,
        )

    with rio.open(pairs[0][1]) as m_src:
        img2 = m_src.read()
        ep.plot_rgb(
            img2,
            rgb=(0, 3, 2),
            figsize=(10, 10),
            str_clip=2,
            ax=ax5[1],
            extent=None,
            title="MODIS transformed",
            stretch=stretch,
        )

    plt.show()


def plot_clipped_modis_lsat():
    pairs = util.get_landsat_modis_pairs_early(
        util.OUTPUT_DIR+"/landsat_modis_pairs")

    fig6, ax6 = plt.subplots(1, 2)
    ax6[0] = plot_rgb(pairs[0][0],
                      rgb=[3, 2, 1],
                       title="Landsat")

    ax6[1] = plot_rgb(pairs[0][1],
             rgb=[1, 4, 3],
             title="MODIS transformed")
