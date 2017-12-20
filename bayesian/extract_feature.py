#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from pre_process import separate_by_class
import math

__author__ = 'LIU Lihao'


# math basic
def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    var = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers))

    return math.sqrt(var)


# every col's means and stdev
def summarize(dataset):
    summaries = [(mean(col), stdev(col)) for col in zip(*dataset)]
    del summaries[-1]

    return summaries


def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}

    for class_value, class_dataset in separated.items():
        summaries[class_value] = summarize(class_dataset)

    return summaries
