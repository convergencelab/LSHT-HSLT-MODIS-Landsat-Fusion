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
from rioxarray import exceptions as rio_x
import shutil
from tqdm import tqdm

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

def sort(landsat_dir, modis_dir, index):
    ### Part 3: sort files into associated landsat-modis pairs ###
    dir, index = util.build_dataset(output_dir=os.environ['LS_MD_PAIRS'],
                       l_dir=landsat_dir,
                       m_dir=modis_dir,
                       stacked_bands=[1, 2, 3, 4, 5, 6, 7],
                        index=index)
    return dir, index

def affine_transform(dir):
    ### Part 4: apply affine transform to each pair ###
    for path in tqdm(util.get_landsat_modis_pairs_early(dir)):
        try:
            l_path = path[0]
            m_path = path[1]
            new_f = m_path[:-4]+"_transformed.TIF"
            util.reproject_on_tif(inpath=m_path,
                                  outpath=new_f,
                                  to_copy_from_path=l_path)
        except IndexError:
            print("error transforming: {}".format(path))
            try:
                with open(
                        r"C:\Users\Noah Barrett\Desktop\School\Research 2020\code\super-res\LSHT-HSLT-MODIS-Landsat-Fusion\assets\log.txt",
                        "a") as f:
                    f.write("transform error: {}\n".format(path))
            except:
                pass
    return dir

def clip(dir):
    ### Part 5: apply clipping of modis based on landsat bounding box ###
    for path in tqdm(util.get_landsat_modis_pairs_early(dir)):
        try:
            l_path = path[0]
            m_path = path[1]
            new_f = m_path[:-4] + "_clipped.TIF"
            util.clip_tif_wrt_tif(inpath=m_path,
                                  outpath=new_f,
                                  to_copy_from_path=l_path)
        except:
            print("error transforming: {}".format(path))
            try:
                with open(
                        r"C:\Users\Noah Barrett\Desktop\School\Research 2020\code\super-res\LSHT-HSLT-MODIS-Landsat-Fusion\assets\log.txt",
                        "a") as f:
                    f.write("transform error: {}\n".format(path))
            except:
                pass

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
    for path in tqdm(util.get_landsat_modis_pairs_early(dir)):
        try:

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
        except:
            print("error converting to NPY: {}".format(f_path))
            try:
                with open(
                        r"C:\Users\Noah Barrett\Desktop\School\Research 2020\code\super-res\LSHT-HSLT-MODIS-Landsat-Fusion\assets\log.txt",
                        "a") as f:
                    f.write("transform error: {}\n".format(f_path))
            except:
                pass

#################
# Wrap function #
#################
def wrap(l_dir,
         m_dir,
         call_download=True,
         call_unzip=True,
         call_sort=True,
         call_affine_transform=True,
         call_clip=True,
         call_to_NPY=True):
    """
    wrap pipe functions
    :param l_dir:
    :param m_dir:
    :return:
    """
    l_dirs = glob.glob(l_dir + "\*")
    m_dirs = glob.glob(m_dir + "\*")
    index = 0
    dir = os.environ['LS_MD_PAIRS']

    if call_download:
        print("downloading...")
        downloading(continue_toggle=False)
    if call_unzip or call_sort:
        if call_unzip and call_sort:
            print("unzipping and sorting...")
        elif call_unzip:
            print("unzipping...")
        else:
            print("sorting...")

    for l, m in tqdm(zip(l_dirs, m_dirs)):
        if call_unzip:
                unzip(l, m)
        if call_sort:
                dir, index = sort(l, m, index)

    if call_affine_transform:
        print("transforming...")
        dir = affine_transform(dir)
    if call_clip:
        print("clipping...")
        clip(dir)
    if call_to_NPY:
        print("converting to .NPY...")

        to_NPY(dir)



kwargs = {'l_dir':r"C:\Users\Noah Barrett\Desktop\School\Research 2020\data\super-res\landsat",
         'm_dir':r"C:\Users\Noah Barrett\Desktop\School\Research 2020\data\super-res\MODIS",
         'call_download':True,
         'call_unzip':True,
         'call_sort':True,
         'call_affine_transform':True,
         'call_clip':True,
         'call_to_NPY':True}


wrap(**kwargs)
