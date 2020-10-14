import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pickle
import os
import sys
import math

def loadDic():
    with open (os.path.join('data', 'daily_ratings.pickle'), 'rb') as f:
        ratings = pickle.load(f)
    return ratings

def actualPop(ratings, top_n, date):
    user_activities = {}
    for movie in ratings[date]:
        for user in movie:
            u = user_activities.get(user, -1)
            if u == -1:
                user_activities[u] = [movie]
            else:
                user_activities[u].append(movie)
    return user_activities

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
    print(top_n_keys)
    print(top_n_values)
    return top_n_keys

# evaluations
def precisionAtK(recommended, actual, k): # how many recommended items are rated by user
    count = 0
    total = 0
    for user in actual:
        movies = actual[user]
        for movie in movies:
            total += 1
            if movie in recommended:
                count += 1
    return count / k

def recallAtK(recommended, actual, k): # how many items rated by user are recommended
    count = 0
    for item in actual:
        if item in recommended:
            count += 1
    return count / k

def getMovieTitleById(ids):
    movieNames = []
    for movies in pd.read_csv(os.path.join(sys.path[0], 'data', 'ml-25m', 'movies.csv'), iterator=True, chunksize=1000):
        for item in movies[["movieId", "title"]].values:
            if item[0] in ids:
                movieNames.append(item[1])
    print(movieNames)
    return movieNames


