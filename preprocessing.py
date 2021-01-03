import numpy as np
import pandas as pd
import os
import sys
import pickle
import datetime
from utils import getUserActivities, getMaxR

def convert_time(timestamp):
    date=datetime.datetime.fromtimestamp(
    int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')
    return int(date[0:4]), int(date[5:7]), int(date[8:10])

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
for ratings in pd.read_csv(ratings_path, iterator=True, chunksize=1000):
    for item in ratings[["userId", "movieId", "timestamp"]].values:
        if earliest == -1 or latest == -1:
            earliest = item[2]
            latest = item[2]
        else: 
            if item[2] > latest:
                latest = item[2]
            if item[2] < earliest:
                earliest = item[2]       
print(convert_time(earliest), convert_time(latest))

daily_ratings = {}
rating_count = 0
movie_ratings = {}
movie_count = 0

for ratings in pd.read_csv(ratings_path, iterator=True, chunksize=1000):
    for item in ratings[["userId", "movieId", "timestamp"]].values:
        year, month, day = convert_time(item[2])
        rating_count += 1
        # if year >= 2019 or year <= 2008:
        #     continue #ignore the last incomplete year, and only keep latest 10 years of data
        if datetime.date(year, month, day) > datetime.date(2019, 10, 31) or datetime.date(year, month, day) <= datetime.date(2009, 10, 31):
            continue
        i = (year, month, day)
        if daily_ratings.get(i, -1) == -1:
            daily_ratings[i] = {item[1] : [item[0]]}
        else:
            try:
                daily_ratings[i][item[1]].append(item[0])
            except KeyError:
                daily_ratings[i][item[1]] = [item[0]]

        if movie_ratings.get(item[1], -1) == -1:
            movie_count += 1
            movie_ratings[item[1]] = {i : 1}
        else:
            try:
                movie_ratings[item[1]][i] += 1
            except KeyError:
                movie_ratings[item[1]][i] = 1
# print(daily_ratings[(2018, 12, 31)])
# print(movie_ratings[72998])
print(rating_count)
print(movie_count)
day_count = len(daily_ratings)
print(day_count) 

np.save(os.path.join('data', 'daily_ratings.npy'), daily_ratings)
np.save(os.path.join('data', 'movie_ratings.npy'), movie_ratings)
# np.save(os.path.join('data', 'time_lookup.npy'), time_lookup)