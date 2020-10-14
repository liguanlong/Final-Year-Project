import numpy as np
import pandas as pd
import os
import sys
import pickle

full_data_path = os.path.join(sys.path[0], 'ml-25m')
small_data_path = os.path.join(sys.path[0], 'ml-latest-small')
path = None

while (path == None):
    is_full = input("Use full data? (y/n)")
    if (is_full == "y"):
        path = full_data_path
    elif(is_full == "n"):
        path = small_data_path

ratings_path = os.path.join(path, 'ratings.csv')

movies = dict()
users = dict()

for ratings in pd.read_csv(ratings_path, iterator=True, chunksize=1000):
    for item in ratings[["userId", "movieId", "timestamp"]].values:
        if item[0] not in users:
            users[item[0]] = [(item[1],item[2])]
        else:
            users[item[0]].append((item[1], item[2]))
        if item[1] not in movies:   
            movies[item[1]] = [(item[0], item[2])]
        else:
            movies[item[1]].append((item[0], item[2]))

for m in movies.keys():
    print(m, len(movies[m]))
with open ('movies.pickle', 'wb') as f:
    pickle.dump(movies, f, protocol=pickle.HIGHEST_PROTOCOL)


for u in users.keys():
    print(u, len(users[u]))
with open ('users.pickle', 'wb') as f:
    pickle.dump(users, f, protocol=pickle.HIGHEST_PROTOCOL)