import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import sys
import math
import datetime

def loadDic():
    daily_ratings_path = os.path.join('data', 'daily_ratings.npy')
    daily_ratings = np.load(daily_ratings_path, allow_pickle=True).item()
    # movie_ratings_path = os.path.join('data', 'movie_ratings.npy')
    # movie_ratings = np.load(movie_ratings_path, allow_pickle=True).item()
    # time_lookup_path = os.path.join('data', 'time_lookup.npy')
    # time_lookup = np.load(time_lookup_path, allow_pickle=True).item()
    # return ratings, time_lookup
    return daily_ratings

# prediction 
def mostPop(ratings, top_n, until):
    movies = {}
    for k in ratings:
        k_date = datetime.date(k[0], k[1], k[2])
        until_date = datetime.date(until[0], until[1], until[2])
        if k_date >= until_date:
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
        k_date = datetime.date(k[0], k[1], k[2])
        until_date = datetime.date(until[0], until[1], until[2])
        if k_date >= until_date or (until_date - k_date).days > recent:
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
        k_date = datetime.date(k[0], k[1], k[2])
        until_date = datetime.date(until[0], until[1], until[2])
        if k_date >= until_date or (until_date - k_date).days > recent:
            continue
        how_recent = -(until_date - k_date).days
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
    # print(sum(top_n_values))
    # print(top_n_keys)
    # print(top_n_values)
    return top_n_keys

def getUserActivities(rating):
    user_activities = {}
    for movie in rating:
        for user in rating[movie]:
            if user_activities.get(user, -1) != -1:
                user_activities[user].append(movie)
            else:
                user_activities[user] = [movie]
    # print(len(user_activities))
    return user_activities

def getMaxR(user_activities):
    count = 0
    max_R = 0
    for user in user_activities:
        movies = user_activities[user]
        R = len(movies)
        count += R
        max_R = max(R, max_R)

    # print(user_activities)
    # print(count)
    # print(max_R)
    return max_R

# evaluations
def RPrecision(user_activities, max_R_predicted):
    R_precision = 0
    for user in user_activities:
        movies = user_activities[user]
        R = len(movies)
        top_R = max_R_predicted[0: R]
        r = 0
        
        for movie in movies:
            if movie in top_R:
                r += 1
        R_precision += (r / R)
    R_precision /= len(user_activities)
    return R_precision

def getMovieTitleById(ids):
    movieNames = []
    for movies in pd.read_csv(os.path.join(sys.path[0], 'data', 'ml-25m', 'movies.csv'), iterator=True, chunksize=1000):
        for item in movies[["movieId", "title"]].values:
            if item[0] in ids:
                movieNames.append(item[1])
    print(movieNames)
    return movieNames


# not used in the evaluation
# def precisionAtK(recommended, actual, k): # how many recommended items are rated by user
#     count = 0
#     total = 0
#     for movie in recommended:
#         total += 1
#         for watched in actual:
#             if movie == watched:
#                 count += 1
#                 break
#     return count / total

# def recallAtK(recommended, actual, k): # how many items rated by user are recommended
#     count = 0
#     total = 0
#     watched_set = set(actual)
#     for watched in watched_set:
#         total += 1
#         if watched in recommended:
#             count += 1
#     return count / total

# def ndcgAtK(recommended, actual, k):
#     watched_dict = {}
#     for watched in actual:
#         if watched not in watched_dict:
#             watched_dict[watched] = 1
#         else:
#             watched_dict[watched] += 1
#     dcg = 0
#     rel_list = []
#     for movie in recommended:
#         pos = recommended.index(movie) + 1
#         rel = watched_dict.get(movie, 0)
#         rel_list.append(rel)
#         dcg += rel / math.log2(pos + 1)
#     if dcg == 0:
#         return dcg
#     idcg = 0
#     ipos = 1
#     rel_list.sort(reverse=True)
#     for rel in rel_list:
#         idcg += rel / math.log2(ipos + 1) 
#         ipos += 1
#     return dcg / idcg