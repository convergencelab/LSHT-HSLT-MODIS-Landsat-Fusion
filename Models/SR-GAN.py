from Data_Extraction.loader import landsat_modis_loader as lml
from Data_Extraction import util

loader = lml(util.OUTPUT_DIR)
print(lml._L_width)
