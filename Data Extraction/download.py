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

class downloader():
    def __init__(self, username, password, OUTPUT_DIR):

        self.username = username
        self.password = password
        self.OUTPUT_DIR = OUTPUT_DIR
        self.DOWNLOAD_LIMIT = 2
        self.Datasets = ("LANDSAT_8_C1", "MODIS_MOD09GA_V6")

        ### Time ###
        self._TIME_FRAME = [
                      '2013-04-23',
                      '2020-01-01'
                      ]

        self.TIME_FRAME = [date.fromisoformat(t) for t in self._TIME_FRAME]
        self.TOTAL_DAYS = np.abs((self.TIME_FRAME[0] - self.TIME_FRAME[1]).days)
        self.CUR_DATE = self.TIME_FRAME[0]

        ### initialize landsatxplore ###
        self.EEE = EarthExplorerExtended(username=self.username, password=self.password)

        ### GET LATS-LONS ###
        self.lat_lon = util.load_world_lat_lon(OUTPUT_DIR + r"\world_lat_lon\world_country_and_usa_states_latitude_and_longitude_values.csv")
        self.ll_iter = self.lat_lon.iterrows()

    def download_all(self, continue_toggle=False):
        """
        downloads all scenes found until download limit is met.
        :return: None
        """
        ### SEARCHING EarthExplorer ###
        # Goal: find matching scenes for both data sets
        while True:
            if os.path.isdir(os.path.join(self.OUTPUT_DIR + "/landsat", location.country)) and \
                os.path.isdir(os.path.join(self.OUTPUT_DIR + "/MODIS", location.country)):
                    continue
            # Loop to find lats/lons that will work
            # breaks when finds first match
            index, location = next(self.ll_iter)
            lat = location.latitude
            lon = location.longitude
            print("Searching for items at {}, {}".format(lat, lon))
            scenes = self.EEE.GET_MODIS_LANDSAT_PAIRS(datasets=self.Datasets,
                                                latitude=lat,
                                                longitude=lon,
                                                start_date=str(self.TIME_FRAME[0]),
                                                end_date=str(self.TIME_FRAME[1]),
                                                max_cloud_cover=10
                                                 )
            if not len(scenes):
                print("No scenes found for {} in {} and {}".format(location.country, self.Datasets[0], self.Datasets[1]))

            else:
                print('{} scenes found for {} and {} in {}.'.format(len(scenes), self.Datasets[0], self.Datasets[1], location.country))

                ### Download ###
                try:
                    os.mkdir(os.path.join(self.OUTPUT_DIR + "/landsat", location.country))
                    os.mkdir(os.path.join(self.OUTPUT_DIR + "/MODIS", location.country))

                except FileNotFoundError:
                    os.mkdir(self.OUTPUT_DIR + "/landsat")
                    os.mkdir(self.OUTPUT_DIR + "/MODIS")
                    os.mkdir(os.path.join(self.OUTPUT_DIR + "/landsat", location.country))
                    os.mkdir(os.path.join(self.OUTPUT_DIR + "/MODIS", location.country))

                except FileExistsError:
                    pass

                L_dir = os.path.join(self.OUTPUT_DIR + "/landsat", location.country)
                M_dir = os.path.join(self.OUTPUT_DIR + "/MODIS", location.country)
                written_scenes = []
                for i, scene in enumerate(scenes):
                    # download modis and landsat data
                    self.EEE.generic_download(self.Datasets[0], scene[0], L_dir)
                    self.EEE.generic_download(self.Datasets[1], scene[1], M_dir)
                    written_scenes.append(written_scenes)
                    # break after reaching download limit
                    if i == self.DOWNLOAD_LIMIT -1:
                        break

            util.write_to_json(scenes, self.Datasets, self.OUTPUT_DIR)
            # break after first iter for now...
            # this download will still be on the range of 20-25 gb due to the size of landsat images
            # break

            # toggle
            if continue_toggle:
                while True:
                    breaker = input("Continue downloading? (Y/N): ").lower()
                    if breaker == "y" or breaker == "n":
                        break
                    else:
                        print("y or n")
                if breaker == "y":
                    continue
                else:
                    # return dirs where files were downloaded
                    return L_dir, M_dir




