#!/usr/bin/env python
# -*- coding: utf-8 -*-
from predict import *

__author__ = 'LH Liu'


def main():
    filename = './data/pima-indians-diabetes.csv'
    split_ratio = 0.67
    dataset = load_csv(filename)
    training_set, test_set = split_dataset(dataset, split_ratio)
    # prepare model
    summaries = summarize_by_class(training_set)
    # test model
    predictions = predict_all(summaries, test_set)
    accuracy = get_accuracy(test_set, predictions)
    print(accuracy)


if __name__ == '__main__':
    main()
