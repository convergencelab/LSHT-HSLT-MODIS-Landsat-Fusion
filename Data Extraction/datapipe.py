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


def downloading():
    ### Part 1: download images ###
    username = util.USERNAME
    password = util.PASSWORD
    OUTPUT_DIR = util.OUTPUT_DIR

    # intialize downloader
    downloader = download.downloader(username = util.USERNAME,
                                    password = util.PASSWORD,
                                    OUTPUT_DIR = util.OUTPUT_DIR)

    # download set of scenes:
    landsat_dir, modis_dir = downloader.download_all(continue_toggle=True)
    return landsat_dir, modis_dir

def unzip(landsat_dir, modis_dir):
    ### Part 2: unzip and convert files to .tif format ###

    # Landsat needs to be unzipped
    # the below function unzips every tar.gz file in dir
    util.unzip_targz(landsat_dir)

    # MODIS needs to be converted from .hdf to .tif
    util.hdf_to_TIF(modis_dir)

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


def wrap(new_download=True):
    if new_download:
        landsat_dir, modis_dir = downloading()
    else:
        # latest_country = input("latest country: " )
        latest_country = "Andorra"
        landsat_dir = util.OUTPUT_DIR + "/landsat/"+ latest_country
        modis_dir = util.OUTPUT_DIR + "/MODIS/" + latest_country
    print("download complete")
    os.system("PAUSE")
    unzip(landsat_dir, modis_dir)
    print("unzip complete")
    os.system("PAUSE")
    dir = sort(landsat_dir, modis_dir)
    print("sort complete")
    os.system("PAUSE")
    dir = affine_transform(dir)
    print("transform complete")
    os.system("PAUSE")
    print("clip complete")
    clip(dir)

wrap(new_download=False)

def wrap_no_io():
    landsat_dir, modis_dir = downloading()
    unzip(landsat_dir, modis_dir)
    dir = sort(landsat_dir, modis_dir)
    dir = affine_transform(dir)
    clip(dir)