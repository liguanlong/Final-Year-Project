import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pickle
import os
import sys
import math

def loadDic():
    daily_ratings_path = os.path.join('data', 'daily_ratings.npy')
    ratings = np.load(daily_ratings_path, allow_pickle=True).item()
    time_lookup_path = os.path.join('data', 'time_lookup.npy')
    time_lookup = np.load(time_lookup_path, allow_pickle=True).item()
    return ratings, time_lookup

def actualPop(ratings, time_lookup, top_n, date):
    user_activities = {}
    for movie in ratings[date]:
        for user in ratings[date][movie]:
            timestamp = time_lookup[(user,movie)]
            exist = user_activities.get(user, -1)
            if exist == -1:
                user_activities[user] = (movie, timestamp)
            elif timestamp < user_activities[user][1]:
                user_activities[user] = (movie, timestamp)
    for user in user_activities:
        if user_activities[user][1] < 789652009 + 60 * 60 * 24 * date or user_activities[user][1] >= 789652009 + 60 * 60 * 24 * (date + 1):
            print("timestamp out of range for actualPop")
    return [user_activities[user][0] for user in user_activities]

# prediction 
def mostPop(ratings, top_n, until):
    movies = {}
    for k in ratings:
        if k >= until:
            continue
        for m in ratings[k]:
            users = list(ratings[k][m])
            movie = movies.get(m, -1)
            if movie == -1:
                movies[m] = len(users)
            else:
                movies[m] += len(users)
    keys = getTopN(movies, top_n)
    return keys

def recentPop(ratings, top_n, until, recent):
    movies = {}
    for k in ratings:
        if k >= until or k <= until - recent * 30:
            continue
        for m in ratings[k]:
            users = list(ratings[k][m])
            movie = movies.get(m, -1)
            if movie == -1:
                movies[m] = len(users)
            else:
                movies[m] += len(users)
    keys = getTopN(movies, top_n)
    return keys

def decayPop(ratings, top_n, until, recent):
    movies = {}
    for k in ratings:
        if k >= until or k <= until - recent * 30:
            continue
        how_recent = (until - k) // 30 + 1
        weight = math.e ** how_recent
        for m in ratings[k]:
            users = list(ratings[k][m])
            movie = movies.get(m, -1)
            if movie == -1:
                movies[m] = len(users) * weight
            else:
                movies[m] += len(users) * weight
    keys = getTopN(movies, top_n)
    return keys

def getTopN(interactions, top_n):
    top_n_keys = []
    top_n_values = []
    
    for k in interactions:
        if top_n_keys == []:
            top_n_keys = [-1 for i in range(0, top_n)]
        if top_n_values == []:
            top_n_values = [-1 for i in range(0, top_n)]
        index = -1
        for i in range (0, top_n):
            if top_n_values[i] <= interactions[k]:
                index = i
                break
        if index != -1:
            for i in range (top_n - 2, index - 1, -1):
                top_n_keys[i + 1] = top_n_keys[i]
                top_n_values[i + 1] = top_n_values[i]
            top_n_keys[index] = k
            top_n_values[index] = interactions[k]
    # print(top_n_keys)
    # print(top_n_values)
    return top_n_keys

# evaluations
def precisionAtK(recommended, actual, k): # how many recommended items are rated by user
    count = 0
    total = 0
    for movie in recommended:
        total += 1
        for watched in actual:
            if movie == watched:
                count += 1
                break
    return count / total

def recallAtK(recommended, actual, k): # how many items rated by user are recommended
    count = 0
    total = 0
    watched_set = set(actual)
    for watched in watched_set:
        total += 1
        if watched in recommended:
            count += 1
    return count / total

def ndcgAtK(recommended, actual, k):
    watched_dict = {}
    for watched in actual:
        if watched not in watched_dict:
            watched_dict[watched] = 1
        else:
            watched_dict[watched] += 1
    watched_list = [(k,v) for k,v in sorted(watched_dict.items(), key=lambda item:item[1], reverse=True)]
    dcg = 0
    for movie in recommended:
        pos = recommended.index(movie) + 1
        for watched in watched_list:
            if watched[0] == movie:
                rel = watched[1]
                dcg += rel / math.log2(pos + 1)
                break
    idcg = 0
    for watched in watched_list:
        if watched_list.index(watched) >= k:
            break
        ipos = watched_list.index(watched) + 1
        rel = watched[1]
        idcg += rel / math.log2(ipos + 1)
    return dcg / idcg

def getMovieTitleById(ids):
    movieNames = []
    for movies in pd.read_csv(os.path.join(sys.path[0], 'data', 'ml-25m', 'movies.csv'), iterator=True, chunksize=1000):
        for item in movies[["movieId", "title"]].values:
            if item[0] in ids:
                movieNames.append(item[1])
    print(movieNames)
    return movieNames


