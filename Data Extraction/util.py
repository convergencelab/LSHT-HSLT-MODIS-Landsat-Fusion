"""
helper functions
"""
import csv
import pandas as pd
import random
import json

def load_world_lat_lon(file_path):
    df = pd.read_csv(file_path)
    return df

def load_json(file_path):
    f = open(file_path)
    return json.load(f)
