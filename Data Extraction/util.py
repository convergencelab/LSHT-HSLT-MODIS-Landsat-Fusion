"""
helper functions
"""
import csv
import pandas as pd
import random
import json
from datetime import date, timedelta

def load_world_lat_lon(file_path):
    df = pd.read_csv(file_path)
    return df

def load_json(file_path):
    f = open(file_path)
    return json.load(f)
'2020-01-01'
def day_time_step(date):
    date = date.fromisoformat('2019-12-04')
    date += timedelta(days=1)
    return str(date)