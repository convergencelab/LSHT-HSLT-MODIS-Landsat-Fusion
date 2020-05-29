"""
Author: Noah Barrett
Date: 2020-05-29
keeping track of findings:
"""
import util
import visualize
import numpy as np
"""
section 1: downloading data

after getting the downloader working some investigating
will help to get a better understanding of the data

when finding data, I save all potential sets that meet the
requirements of date and location in a json file. 

we can load this metadata using util. 
"""
polygons = util.get_spatial_polygons()
print("CHECK 1: location")
for landsat_modis_pair in polygons:
    print(landsat_modis_pair[0].intersects(landsat_modis_pair[1]))

visualize.plot_spatial_footprints(polygons)

"""
This shows that all polygons intersect,

From the plot you can see that landsat is completely inside the modis image.

This is good as it confirms that the prospective scenes overlap 
between pairs

next we check to see that the dates are allgining up for all pairs: 
"""
dates = np.array(util.get_dates())
landsat = dates.T[0]
modis = dates.T[1]
print("CHECK 2: dates")
print(landsat == modis)

"""
dates all line up as well, so we can confirm that the scraper succesfully collected
pairs that allign both temporally and geographically
************************************************************************************************
Next we will look at cloud cover quality,

For this we only observe landsat, this is because modis data does not actually provide this metadata
this entire project is following the model: prioritize landsat and then find matching modis data
so for this we will do the same thing. Check landsat quality of cloud cover and assume a fairly similar
occurence for modis (same clouds will be present for both images if they were taking in relatively 
same time period
"""
print("CHECK 3: cloud indexes")
landsat_cloud_indexes = util.get_cloud_indexes()
print(landsat_cloud_indexes)
### plot ###
visualize.plot_cloud_cover(landsat_cloud_indexes)

"""
we can see that it has a very good quality for cloud cover (as indicated)
Nothing over 10% and the bulk is roughly between 3-6% so pretty good

This project does not require to totally eliminate cloud coverage because its goal is to 
enhance modis images, most of which will include alot of cloud coverage. It is interesting to look at scenes with less 
clouds however in terms of practicality this should not be the main focus of the project. 

To wrap up this section: 
    - The locations intersect so we will be able match up locations in the found scenes
    - The dates are all the same so we know that the daily phenomena will also be fairly alligned
    - The cloud coverage of landsat images was able to grab the percentage I indicated (10% or less)
        -> this project will not be heavily focused on ensuring extremely low cloud coverage. 

"""