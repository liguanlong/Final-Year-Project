import numpy as np
import pandas as pd
import os
import sys
import pickle
import csv

full_data_path = os.path.join(sys.path[0], 'data', 'ml-25m')
small_data_path = os.path.join(sys.path[0], 'data', 'ml-latest-small')
path = None

while (path == None):
    is_full = input("Use full data? (y/n)")
    if (is_full == "y"):
        path = full_data_path
    elif(is_full == "n"):
        path = small_data_path

ratings_path = os.path.join(path, 'ratings.csv')
earliest = -1
latest = -1

times = []
for ratings in pd.read_csv(ratings_path, iterator=True, chunksize=1000):
    for item in ratings[["userId", "movieId", "timestamp"]].values:
        times.append(item[2])

times.sort()

df = pd.DataFrame(times, columns = ["column"])
df.to_csv('timestamp.scv', index=False)