# LSHT-HSLT-MODIS-Landsat-Fusion
Super-resolution approach to Modis-Landsat RS data Fusion

# section 1: Data Extraction
Download data directly from USGS EarthExplorer API 
Currently only supports particular MODIS and Landsat products, can easily be extended by editing Datasets.json:
supported Datasets extending Landsatxplorer can be found in that document as well. 

Modis-Landsat pairs are downloaded from the USGS EarthExplorer API, there are matched up based on their Aquistition date as well as
their location. 

These pairs are converted to .npy pairs for super-resolution: (landsat(high-res), modis(low-res))

# section 2: exploring models
Using the above downloaded data, test different models:
*  SR-GAN

