"""
Author: Noah Barrett
This project requires MODIS-Landsat pairs

This script is built to download these pairs using pyModis and landsatexplore,
these libraries interact with the NASA and EarthExplorer apis respectively.

Workflow: search for good landsat images, for the existing corresponding modis images,
download both pairs

LANDSAT: We will look for quality level 9 images, with cloud cover of 30% to ensure
we are dealing with fairly decent images.

scene locations are currently based on dataset aquired from:
https://www.kaggle.com/paultimothymooney/latitude-and-longitude-for-every-country-and-state

further implementations will consider more than these lat, lons
"""

import landsatxplore as le
from landsatxplore.earthexplorer import EarthExplorer
import os
import util

"""
intialize api
"""
try:
    username = os.environ['EE_USERNAME']
    password = os.environ['EE_PASSWORD']
except KeyError:
    os.environ['EE_USERNAME'], os.environ['EE_PASSWORD'] = input("username:"), input("pw:")
    username = os.environ['EE_USERNAME']
    password = os.environ['EE_PASSWORD']


EE = EarthExplorer(username=username, password=password)
api = le.api.API(username=username, password=password)

def download(scenes, dir, EE=EE):
    """
    download given scenes
    :param scenes:list of scene objs
    :return: None
    """
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass

    for scene in scenes:
        #print(scene['acquisitionDate'])
        EE.download(scene_id=scene['entityId'], output_dir=dir)
        break# for now break after first scene


"""
Globals
"""
OUTPUT_DIR = "C:/Users/Noah Barrett/Desktop/School/Research 2020/data"
TIME_FRAME = ('2000-01-01', '2020-01-01')
### initialize landsatxplore ###

"""
landsatxplore has two functions:
      download:  Download one or several Landsat scenes.
      search:    Search for Landsat scenes.
      
this script will search for the appropriate images and then download ones
which exist with modis

the quality ensurance of images in this approach will be purely through landsat
images. 
"""
### GET LATS-LONS ###
lat_lon = util.load_world_lat_lon(r"C:\Users\Noah Barrett\Desktop\School\Research 2020\data\world_lat_lon\world_country_and_usa_states_latitude_and_longitude_values.csv")
iter = lat_lon.iterrows()
#iter = random.shuffle(iter)

### SEARCHING EarthExplorer ###
# Goal: find matching scenes for both data sets
data_exists = 0
while not data_exists:
    # Loop to find lats/lons that will work
    index, location = next(iter)
    lat = location.latitude
    lon = location.longitude
    L_scenes = api.search(
                            dataset='LANDSAT_8_C1',
                            latitude=lat,
                            longitude=lon,
                            start_date=TIME_FRAME[0],
                            end_date=TIME_FRAME[1],
                            max_cloud_cover=10)

    """
    MOD09 (MODIS Surface Reflectance) is a seven-band product computed from the MODIS Level 1B land
    bands 1 (620-670 nm), 2 (841-876 nm), 3 (459-479), 4 (545-565 nm), 5 (1230-1250 nm), 6 (1628-1652 nm), and
    7 (2105-2155 nm). The product is an estimate of the surface spectral reflectance for each band as it would have
    been measured at ground level as if there were no atmospheric scattering or absorption. It corrects for the effects
    of atmospheric gases and aerosols. (src. http://modis-sr.ltdri.org/guide/MOD09_UserGuide_v1.4.pdf)
    """

    M_scenes = api.search(
                          dataset='EMODIS',
                          latitude=lat,
                          longitude=lon,
                          max_cloud_cover=10,
                          start_date=TIME_FRAME[0],
                          end_date=TIME_FRAME[1])



    if not len(L_scenes):
        print("No scenes found for {} in Landsat8".format(location.country))
    if not len(M_scenes):
        print("No scenes found for {} in MODIS".format(location.country))
    else:
        print('{}, {} scenes found for {} in Landsat8 and MODIS.'.format(len(L_scenes), len(M_scenes), location.country))
        data_exists = 1




### Download ###
L_dir = os.path.join(OUTPUT_DIR+"/landsat", location.country)
M_dir = os.path.join(OUTPUT_DIR+"/MODIS", location.country)
download(L_scenes, L_dir)
download(M_scenes, M_dir)
api.logout()