#!/usr/bin/env python
# -*- coding: utf-8 -*-
from extract_feature import *

__author__ = 'LH Liu'


def calculate_probability(x, mean_x, stdev_x):
    exponent = math.exp(-(((x - mean_x) ** 2) / (2 * (stdev_x ** 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev_x)) * exponent


def calculate_class_probabilities(summaries, input_vec):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        for i in range(len(class_summaries)):
            if class_value not in probabilities:
                probabilities.setdefault(class_value, 1)
            mean_class, stdev_class = class_summaries[i][0]
            x = input_vec[i]
            probabilities[class_value] *= calculate_probability(x, mean_class, stdev_class)
    return probabilities


def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_label = class_value
            best_prob = probability
    # best_prob = reduce(lambda x, y: x if x > y else y , probabilities.values())
    return best_label


def predict_all(summaries, input_vectors):
    predictions = []
    for i, vec in enumerate(input_vectors):
        best_lebal = predict(summaries, vec)
        predictions.append(best_lebal)
    return predictions


def get_accuracy(test_set, predictions):
    correct = 0
    for i, vec in enumerate(test_set):
        if vec[-1] == predictions[i]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0
