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

***Notes:
MOD09 (MODIS Surface Reflectance) is a seven-band product computed from the MODIS Level 1B land
bands 1 (620-670 nm), 2 (841-876 nm), 3 (459-479), 4 (545-565 nm), 5 (1230-1250 nm), 6 (1628-1652 nm), and
7 (2105-2155 nm). The product is an estimate of the surface spectral reflectance for each band as it would have
been measured at ground level as if there were no atmospheric scattering or absorption. It corrects for the effects
of atmospheric gases and aerosols. (src. http://modis-sr.ltdri.org/guide/MOD09_UserGuide_v1.4.pdf)
***
"""
import landsatxplore as le
from landsatxplore.earthexplorer import EarthExplorer
from EE_api_extension import EarthExplorerExtended
import os
import util
from datetime import date, timedelta
import numpy as np


Username = util.USERNAME
Password = util.PASSWORD
OUTPUT_DIR = util.OUTPUT_DIR

### initialize landsatxplore ###
EEE = EarthExplorerExtended(username=username, password=password)

### GET LATS-LONS ###
lat_lon = util.load_world_lat_lon(OUTPUT_DIR + r"\world_lat_lon\world_country_and_usa_states_latitude_and_longitude_values.csv")
ll_iter = lat_lon.iterrows()

### Params ###
Datasets = ("LANDSAT_8_C1", "MODIS_MOD09GA_V6")

TIME_FRAME = [
              '2013-04-23',
              '2020-01-01'
              ]

### convert to datetime obj ###
TIME_FRAME = [date.fromisoformat(t) for t in TIME_FRAME]
TOTAL_DAYS = np.abs((TIME_FRAME[0] - TIME_FRAME[1]).days)
CUR_DATE = TIME_FRAME[0]

### SEARCHING EarthExplorer ###
# Goal: find matching scenes for both data sets
while True:
    # Loop to find lats/lons that will work
    # breaks when finds first match
    index, location = next(ll_iter)
    lat = location.latitude
    lon = location.longitude
    print("Searching for items at {}, {}".format(lat, lon))
    scenes = EEE.GET_MODIS_LANDSAT_PAIRS(datasets=Datasets,
                                        latitude=lat,
                                        longitude=lon,
                                        start_date=str(TIME_FRAME[0]),
                                        end_date=str(TIME_FRAME[1]),
                                        max_cloud_cover=10,
                                         num_pairs=2
                                         )
    if not len(scenes):
        print("No scenes found for {} in {} and {}".format(location.country, Datasets[0], Datasets[1]))

    else:
        print('{} scenes found for {} and {} in {}.'.format(len(scenes), Datasets[0], Datasets[1], location.country))
        util.write_to_json(scenes, Datasets, OUTPUT_DIR)
        ### Download ###
        try:
            os.mkdir(os.path.join(OUTPUT_DIR + "/landsat", location.country))
            os.mkdir(os.path.join(OUTPUT_DIR + "/MODIS", location.country))
        except FileExistsError:
            pass
        L_dir = os.path.join(OUTPUT_DIR + "/landsat", location.country)
        M_dir = os.path.join(OUTPUT_DIR + "/MODIS", location.country)

        #for scene in scenes:

           # EEE.generic_download(Datasets[1], scene[1], M_dir)
            #EEE.generic_download(Datasets[0], scene[0], L_dir)


            # break after first iter for now...
           # break
        # break after first iter for now...
        break


#api.logout()