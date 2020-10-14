import numpy as np
import pandas as pd
import os
import sys
import pickle

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
print(earliest, latest)

num_of_days = (latest - earliest) // (60 * 60 * 24) #rounded down
print(num_of_days)

rounding = 60 * 60 #round the timestamp
timeranges = {}
timerange_exist = {}
for i in range (0, num_of_days):
    low = earliest + i * 60 * 60 * 24
    high = earliest + (i + 1) * 60 * 60 * 24
    for t in range (low, high):
        t = ((t - earliest) // rounding * rounding) + earliest
        timeranges[t] = i
    timerange_exist[i] = False
print(len(timeranges))

daily_ratings = {}
rating_count = 0
item_count = 0
for ratings in pd.read_csv(ratings_path, iterator=True, chunksize=1000):
    for item in ratings[["userId", "movieId", "timestamp"]].values:
        t = ((item[2] - earliest) // rounding * rounding) + earliest
        rating_count += 1
        if t >= 1574250409:
            continue #ignore the last incomplete 24 hours
        i = timeranges[t]
        if not timerange_exist[i]:
            daily_ratings[i] = {item[1] : [item[0]]}
            timerange_exist[i] = True
        else:
            try:
                daily_ratings[i][item[1]].append(item[0])
            except KeyError:
                daily_ratings[i][item[1]] = [item[0]]
        item_count += 1

print(rating_count)
print(len(timerange_exist))
day_count = len(daily_ratings)
print(day_count) 

with open (os.path.join('data', 'daily_ratings.pickle'), 'wb') as f:
    pickle.dump(daily_ratings, f, protocol=pickle.HIGHEST_PROTOCOL)