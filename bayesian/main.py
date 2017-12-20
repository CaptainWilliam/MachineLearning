#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from pre_process import load_csv, split_dataset
from predict import summarize_by_class, predict_all, get_accuracy

__author__ = 'LIU Lihao'


def main():
    filename = './data/pima-indians-diabetes.csv'
    dataset = load_csv(filename)

    split_ratio = 0.67
    training_set, test_set = split_dataset(dataset, split_ratio)

    # prepare model
    summaries = summarize_by_class(training_set)

    # test model
    predictions = predict_all(summaries, test_set)
    accuracy = get_accuracy(test_set, predictions)

    print accuracy


if __name__ == '__main__':
    main()
