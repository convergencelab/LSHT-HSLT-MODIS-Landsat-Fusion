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


### Part 1: download images ###
username = util.USERNAME
password = util.PASSWORD
OUTPUT_DIR = util.OUTPUT_DIR

# intialize downloader
downloader = download.downloader(username = util.USERNAME,
                                password = util.PASSWORD,
                                OUTPUT_DIR = util.OUTPUT_DIR)

# download set of scenes:
landsat_dir, modis_dir = downloader.download_all()

### Part 2: unzip and convert files to .tif format ###

# Landsat needs to be unzipped
# the below function unzips every tar.gz file in dir
util.unzip_targz(landsat_dir)

# MODIS needs to be converted from .hdf to .tif
util.hdf_to_TIF(modis_dir)

# after this function call is done organize the dir.
util.organize_dir(modis_dir)

### Part 3: sort files into associated landsat-modis pairs ###
dir = util.build_dataset(output_dir=os.environ['LS_MD_PAIRS'],
                   l_dir=landsat_dir,
                   m_dir=modis_dir,
                   stacked_bands=[1, 2, 3, 4, 5, 6, 7])


### Part 4: apply affine transform to each pair ###
for path in util.get_landsat_modis_pairs(dir, transform=False):
    l_path= path[0]
    m_path = path[1]
    new_f = m_path[:-4]+"_transformed.TIF"
    util.reproject_on_tif(inpath=m_path,
                          outpath=l_path,
                          to_copy_from_path=new_f)