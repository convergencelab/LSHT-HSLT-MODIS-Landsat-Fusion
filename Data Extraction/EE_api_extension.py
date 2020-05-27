"""
Extension to landxplorer api
allows for extraction of more than landsat data from usgs

purpose for this project is to get modis data sets

"""

# import requests
import landsatxplore as le
from landsatxplore.earthexplorer import EarthExplorer

### SEARCH DOWNLOADS ###
def search_all(api, datasetName, startDate, endDate, lat, lon, bbox=[10,10]):
    """
    search api for any given data set
    *** adding this as landsatxplorer does not extend well to other
        datasets. ***
    :param apikey: generated upon login
    :param datasetName: Name of dataset to be searched
    :param startDate: Start date for search
    :param endDate: end date for search
    :param lat: center lat param
    :param lon: center lon param
    :param endpoint: api endpoint
    :param bbox: bounding box around center coords
    :return: resulting scenes
    """
    # API endpoint
    #URL = endpoint + 'datasets'
    # TODO: Determine why errors finding MODIS (Bounding boxes)
    # with current defaulted settings matching up on bahamas for both.
    PARAMS = {
              "datasetName":datasetName,
              "publicOnly":False,
             # "spatialFilter": {"filterType": "mbr",
              #                  "lowerLeft": {"latitude":lat-(bbox[0]/2),
             #                                 "longitude": lon-(bbox[1]/2)},
              #                  "upperRight": {"latitude": lat+(bbox[0]/2),
             #                                  "longitude": lon+(bbox[1]/2)}},
            #  "temporalFilter":{"startDate":startDate,"endDate":endDate}
              }
    data = api.request('search', **PARAMS)
    # sending get request and saving the response as response object
    #r = requests.get(url=URL, params=PARAMS)
    #print(r.url)
    # extracting data in json format
    #data = r.json()

    return data
"""
testing


#EE = EarthExplorer(username=username, password=password)
api = le.api.API(username=input("username: "), password=input("password: "))

KEY, ENDPOINT = api.key, api.endpoint

test = search_all(
            api=api,
           datasetName="EMODIS",
           startDate='2000-01-01',
           endDate='2020-01-01',
           lat=10,
           lon=0,
           bbox=[10,10])

print(test)
api.logout()
"""