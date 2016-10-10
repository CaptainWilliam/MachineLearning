# !/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import unicode_literals
import codecs
import math


# ----------------------------------------------------------------------------------------------------------------------
# return format: {user:{movie_1:score, movie_2:score}}
def get_user_prefection_data():
    user_prefs = {}
    movie_data = get_enum_value_for_item_id()
    with codecs.open('E:\PythonProject\Lazy\lazy_1\collaborative_filtering\data\\u.data', mode='r',
                     encoding='utf-8') as prefs_data_reader:
        for line in prefs_data_reader:
            user_id, movie_id, score, _ = line.strip().split('\t')
            if user_id not in user_prefs:
                user_prefs.setdefault(user_id, {})
            user_prefs.get(user_id).setdefault(movie_data.get(movie_id), float(score))
    return user_prefs


# return format: {movie:{user_1:score, user_2:score}}
def get_movie_prefection_data():
    movie_prefs = {}
    movie_data = get_enum_value_for_item_id()
    with codecs.open('E:\PythonProject\Lazy\lazy_1\collaborative_filtering\data\\u.data', mode='r',
                     encoding='utf-8') as prefs_data_reader:
        for line in prefs_data_reader:
            user_id, movie_id, score, _ = line.strip().split('\t')
            if movie_id not in movie_prefs:
                movie_prefs.setdefault(movie_data.get(movie_id), {})
            movie_prefs.get(movie_data.get(movie_id)).setdefault(user_id, float(score))
    return movie_prefs


def get_enum_value_for_item_id():
    movie_data = {}
    with codecs.open('E:\PythonProject\Lazy\lazy_1\collaborative_filtering\data\\u.item', mode='r',
                     encoding='utf-8') as movie_data_reader:
        for line in movie_data_reader:
            movie_id, movie_name = line.split('|')[0:2]
            movie_data.setdefault(movie_id, movie_name)
    return movie_data
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# func 1(also could used for calculate sim for items)
def sim_distance(prefs, user_id_1, user_id_2):
    si = {}
    for movie_name in prefs[user_id_1]:
        if movie_name in prefs[user_id_2]:
            si.setdefault(movie_name, 1)
    # no same interest
    if len(si) == 0:
        return 0
    # calculate distance
    sum_of_squares = sum([(prefs[user_id_1][movie] - prefs[user_id_2][movie]) ** 2 for movie, _ in si.items()])
    return 1 / (1 + math.sqrt(sum_of_squares))


# func 2(also could used for calculate sim for items)
def sim_pearson(prefs, user_id_1, user_id_2):
    si = {}
    for item in prefs[user_id_1]:
        if item in prefs[user_id_2]:
            si.setdefault(item, 1)
    if len(si) == 0:
        return 0
    # begin calculate
    n = len(si)
    sum_1 = sum([prefs[user_id_1][item] for item, _ in si.items()])
    sum_2 = sum([prefs[user_id_2][item] for item, _ in si.items()])
    sum_sq_1 = sum([prefs[user_id_1][item] ** 2 for item, _ in si.items()])
    sum_sq_2 = sum([prefs[user_id_2][item] ** 2 for item, _ in si.items()])
    p_sum = sum([prefs[user_id_1][item] * prefs[user_id_2][item] for item, _ in si.items()])
    num = p_sum - (sum_1 * sum_2 / n)
    den = math.sqrt((sum_sq_1 - (sum_1 ** 2) / n) * (sum_sq_2 - (sum_2 ** 2) / n))
    # end of calculateion
    if den == 0:
        return 0
    return num / den
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# both fit for get similar user or movie
def top_n_matches(prefs, user_id, n=5, similarity=sim_distance):
    scores = [(similarity(prefs, user_id, other), other) for other in prefs if other != user_id]
    scores.sort(reverse=True)
    return scores[0:n]


# get similar movie dataset
def top_n_similar_movie_for_all_movies(prefs, n=10):
    result={}
    for item, _ in prefs.items():
        # pref use get_movie_prefection_data()
        scores = top_n_matches(prefs, item, n=n, similarity=sim_distance)
        result.setdefault(item, scores)
    return result


# get similar user dataset
def top_n_similar_user_for_all_users(prefs, n=10):
    result={}
    for user, _ in prefs.items():
        # pref use get_user_prefection_data()
        scores = top_n_matches(prefs, user, n=n, similarity=sim_distance)
        result.setdefault(user, scores)
    return result
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def get_recommende_items_based_on_items(prefs, item_similar_dataset, user):
    user_rated_item_dataset = prefs[user]
    scores = {}
    total_sim = {}
    # loop over items rated by this user
    for item, score in user_rated_item_dataset.items():
        # loop over items similar to this one
        for similarity, similar_item in item_similar_dataset[item]:
            # ignore the item user already rated
            if similar_item in user_rated_item_dataset:
                continue
            # weighted sum of rating times similarity
            scores.setdefault(similar_item, 0)
            scores[similar_item] += similarity * score
            # sum of all the similarities
            total_sim.setdefault(similar_item, 0)
            total_sim[similar_item] += similarity
            # divide each total score by total weighting to get an average
    rankings = [(score / total_sim[item], item) for item, score in scores.items()]

    # return the rankings from highest to lowest
    rankings.sort(reverse=True)
    return rankings


# 基于用户推荐物品
def get_recommende_items_based_on_users(prefs, user, similarity=sim_pearson):
    totals = {}
    sim_sums = {}
    for other, _ in prefs.items():
        if other == user:
            continue
        # delete the negative related user
        sim = similarity(prefs, user, other)
        if sim <= 0:
            continue
        for item in prefs[other]:
            if item in prefs[user]:
                continue
            totals.setdefault(item, 0)
            totals[item] += sim * prefs[other][item]
            sim_sums.setdefault(item, 0)
            sim_sums[item] += sim
    rankings = [(total / sim_sums[item], item) for item, total in totals.items()]
    rankings.sort(reverse=True)
    return rankings
# ----------------------------------------------------------------------------------------------------------------------


user_prefections = get_user_prefection_data()
movie_prefections = get_movie_prefection_data()
# print top_n_matches(user_prefections, '6')
item_similarity_dataset = top_n_similar_movie_for_all_movies(movie_prefections)
print get_recommende_items_based_on_items(user_prefections, item_similarity_dataset, '6')
print get_recommende_items_based_on_users(user_prefections, '6')
