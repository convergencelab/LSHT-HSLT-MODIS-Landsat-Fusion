"""
Author: Noah Barrett
date: 2020-06-03
Wrapper script to bring all components together:

Consists of 5 parts:
    1) download images
    2) unzip and convert files to .tif format
    3) sort files into associated landsat-modis pairs
    4) apply affine transform to each pair
    5) crop pairs to same dimensions

"""
import util
import download
import os
import glob
import numpy as np
import rasterio as rio


def downloading(continue_toggle=True):
    ### Part 1: download images ###
    username = util.USERNAME
    password = util.PASSWORD
    OUTPUT_DIR = util.OUTPUT_DIR

    # intialize downloader
    downloader = download.downloader(username = util.USERNAME,
                                    password = util.PASSWORD,
                                    OUTPUT_DIR = util.OUTPUT_DIR)

    # download set of scenes:
    landsat_dir, modis_dir = downloader.download_all(continue_toggle=continue_toggle)
    return landsat_dir, modis_dir

def unzip(landsat_dir, modis_dir):
    ### Part 2: unzip and convert files to .tif format ###

    # Landsat needs to be unzipped
    # the below function unzips every tar.gz file in dir
    try:
        util.unzip_targz(landsat_dir)
    except FileExistsError:
        pass

    # MODIS needs to be converted from .hdf to .tif
    try:
        util.hdf_to_TIF(modis_dir)
    except FileExistsError:
        pass
    # after this function call is done organize the dir.
    util.organize_dir(modis_dir)

def sort(landsat_dir, modis_dir):
    ### Part 3: sort files into associated landsat-modis pairs ###
    dir = util.build_dataset(output_dir=os.environ['LS_MD_PAIRS'],
                       l_dir=landsat_dir,
                       m_dir=modis_dir,
                       stacked_bands=[1, 2, 3, 4, 5, 6, 7])
    return dir

def affine_transform(dir):
    ### Part 4: apply affine transform to each pair ###
    for path in util.get_landsat_modis_pairs_early(dir):
        l_path = path[0]
        m_path = path[1]
        new_f = m_path[:-4]+"_transformed.TIF"
        util.reproject_on_tif(inpath=m_path,
                              outpath=new_f,
                              to_copy_from_path=l_path)
    return dir

def clip(dir):
    ### Part 5: apply clipping of modis based on landsat bounding box ###
    for path in util.get_landsat_modis_pairs_early(dir):
        l_path = path[0]
        m_path = path[1]
        new_f = m_path[:-4] + "_clipped.TIF"
        util.clip_tif_wrt_tif(inpath=m_path,
                              outpath=new_f,
                              to_copy_from_path=l_path)

def to_NPY(dir, bands=[[3,2,1], [1, 4, 3]]):
    """
    directory of pairs of landsat modis scenes
    one country at a time
    :param dir: str path to file directory (country)
    :param bands: tuple for bands to be recorded from each tiff, defaulted to rgb
    :return: None
    """

    NPY_dir = os.path.join(util.OUTPUT_DIR, "NPY")
    if not os.path.isdir(NPY_dir):
        os.mkdir(NPY_dir)
    for path in util.get_landsat_modis_pairs_early(dir):

        # landsat and modis pairs #
        l_path = path[0]
        m_path = path[1]

        # get both scene images from file names #
        L_ID = os.path.basename(l_path)[:40]
        M_ID = os.path.basename(m_path)[:27]
        MetaID = np.array([L_ID, M_ID])

        # make file name based on pair and country #
        num = os.path.basename(os.path.split(l_path)[-2])
        fname = num + ".npy"

        f_path = os.path.join(NPY_dir, fname)

        # open landsat #
        l_raster = rio.open(l_path)

        # open MODIS #
        m_raster = rio.open(m_path)
        l_m_bands = [[], []]
        for l_band, m_band in zip(bands[0], bands[1]):
            # read band for both rasters #
            l_m_bands[0].append(l_raster.read(l_band))
            l_m_bands[1].append(m_raster.read(m_band))


        # stack the bands #
        l_stack = np.dstack(l_m_bands[0])
        m_stack = np.dstack(l_m_bands[1])

        ### save as numpy ###

        with open(f_path, 'wb') as f:
            # lsat
            np.save(f, l_stack)
            # modis
            np.save(f, m_stack)
            # include IDs in .npy file so we can
            # keep track of what scenes we are working with
            np.save(f, MetaID)

def wrap(new_download=True):
    if new_download:
        dirs = downloading()
    else:
        # latest_country = input("latest country: " )
        latest_country = "Andorra"
        landsat_dir = glob.glob(util.OUTPUT_DIR + "/landsat/*")
        modis_dir =  glob.glob(util.OUTPUT_DIR + "/MODIS/*")
        dirs = [[l, m] for l, m in zip(landsat_dir, modis_dir)]
    print("download complete")
    os.system("PAUSE")

    for landsat_dir, modis_dir in dirs:
        unzip(landsat_dir, modis_dir)
        print("unzip complete")
        os.system("PAUSE")
        dir = sort(landsat_dir, modis_dir)
        print("sort complete")
        os.system("PAUSE")
        dir = affine_transform(dir)
        print("transform complete")
        os.system("PAUSE")
        clip(dir)
        print("clip complete")
        to_NPY(dir)



def wrap_no_io():
    dirs = downloading(continue_toggle=False)
    for landsat_dir, modis_dir in dirs:
        unzip(landsat_dir, modis_dir)
        dir = sort(landsat_dir, modis_dir)
        dir = affine_transform(dir)
        clip(dir)
        to_NPY(dir)

wrap_no_io()
