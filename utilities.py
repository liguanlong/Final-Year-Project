import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pickle
import os
import sys
import math

def loadDic():
    with open (os.path.join('data', 'movies.pickle'), 'rb') as f:
        movies = pickle.load(f)
    with open (os.path.join('data','users.pickle'), 'rb') as f:
        users = pickle.load(f)
    return (movies, users)

def plotHistogram(item_dic, low_threshold, how_recent, step, if_density, item_name):
    print('plotting for', item_name, low_threshold, how_recent)
    all_interaction = list()
    how_recent = how_recent * 60 * 60 * 24
    zerocount = 0
    norecentcount = 0
    mosts_recent = 0
    
    if item_name == 'movies':
        for k in item_dic.keys():
            local_most_recent = max([item[1] for item in item_dic[k]])
            if local_most_recent > mosts_recent:
                mosts_recent = local_most_recent
    
    for k in item_dic.keys():
        count_interaction = 0
        if how_recent == 0:
            count_interaction = len(item_dic[k])
        else:
            if item_name == 'users':
                mosts_recent = max([item[1] for item in item_dic[k]])
            recent = mosts_recent - how_recent
            for item in item_dic[k]:
                if item[1] >= recent:
                    count_interaction += 1
                else:
                    norecentcount += 1
        
        if count_interaction >= low_threshold:
            all_interaction.append(count_interaction)
        else:
            zerocount += 1
    
    if all_interaction != []:        
        max_interaction_occurance = 0
        max_interaction = max(all_interaction)
        for interaction in all_interaction:
            if interaction == max_interaction:
                max_interaction_occurance += 1
        print('max interaction:', max_interaction)
        print('occurance of max interaction:', max_interaction_occurance)

        bin_range = math.ceil(max_interaction / step)
        bins = []
        for i in range (0, step + 1):
            if item_name == 'users':
                bins.append(low_threshold + i * bin_range)
            if item_name == 'movies':
                bins.append(low_threshold + i * bin_range)
        plt.hist(all_interaction, bins=bins, density=if_density)
        plt.title(item_name)
        plt.xlabel('number of interaction')
        plt.ylabel('number of occurance')
        plt.savefig(item_name + "_threshold" + str(low_threshold) + "_howrecent" + str(how_recent))
        plt.show()
        print('num of non-zero interaction:', len(all_interaction))
        print('num of zero interaction:', zerocount)
        print('num of out-dated interactino', norecentcount)
        print('----------------------------------------------------------')
    return

def recommandBasedOnTotalInteractionForMovie(movies, top_n):
    top_n_keys = []
    top_n_values = []
    interactions = {}
    for k in movies.keys():
        interactions[k] = len(movies[k])
    for k in interactions.keys():
        if top_n_keys == []:
            top_n_keys = [0 for i in range(0, top_n)]
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


def plotHeatMapPopularMoviesPercentageInUserActivityGroups(users, popularMovies, maxActivity, numSeperation):
    step = maxActivity / numSeperation
    data = np.zeros(shape=(10, numSeperation + 1))
    xlabels = []
    for i in range (0, numSeperation + 1):
        low = step * i
        high = step * (i + 1)
        if i == numSeperation:
            xlabels.append(">" + str(low))
        else:
            xlabels.append(str(low) + "-" + str(high))
        userGroup = []
        for k in users.keys():
            if len(users[k]) > low and (i == numSeperation or len(users[k]) <= high):
                userGroup.append(k)
        for k in userGroup:
            popularCount = 0
            for item in users[k]:
                if item[0] in popularMovies:
                    popularCount += 1
            frequency = (popularCount * 10) // len(users[k])
            data[9 - frequency][i] += 1

    fig, ax = plt.subplots()
    sb.heatmap(data, ax=ax, cmap='Reds')
    ax.set_xticklabels(xlabels, rotation = 60)
    ax.set_yticklabels([str((i-1)*10) + "%" + "-" + str(i*10) + "%" for i in range (10, 0, -1)], rotation=0)
    plt.xlabel("number of interaction")
    plt.ylabel("percentage of popular movies")
    plt.tight_layout()
    plt.savefig(os.path.join("figures","heatmap"+ str(len(popularMovies)) ))
    plt.show()
    return