"""
helper functions
"""
import csv
import pandas as pd
import random

def load_world_lat_lon(file_path):
    df = pd.read_csv(file_path)
    return df

