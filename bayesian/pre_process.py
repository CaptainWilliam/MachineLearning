#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import codecs
import random
import csv

__author__ = 'LIU Lihao'


# func_1.load
def load_csv(filename):
    with codecs.open(filename, 'r') as file:
        lines = csv.reader(file)
        dataset = list(lines)

        # double list
        dataset = [[float(x) for x in dataset[i]] for i in range(len(dataset))]

        return dataset


# func_2.random split
def split_dataset(dataset, split_ratio):
    train_size = int(len(dataset) * split_ratio)
    train_set = []
    test_set = list(dataset)

    while len(train_set) < train_size:
        index = random.randrange(len(test_set))
        train_set.append(test_set.pop(index))

    return [train_set, test_set]


# func_3.separate
def separate_by_class(dataset):
    separated = {}

    for i in range(len(dataset)):
        if dataset[i][-1] not in separated:
            separated.setdefault(dataset[i][-1], [])
        separated[dataset[i][-1]].append(dataset[i])

    return separated
