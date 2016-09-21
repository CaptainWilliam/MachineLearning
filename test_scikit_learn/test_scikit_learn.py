from scipy.stats import uniform as sp_rand
from sklearn import preprocessing
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import numpy as np
import urllib

# load data from url
data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
raw_data = urllib.urlopen(data_url)
dataset = np.loadtxt(raw_data, delimiter=',')
input_dataset = dataset[:, :-1]
print len(input_dataset[0])
input_dataset_2 = dataset[:, :8]
print len(input_dataset_2[0])
output_classification = dataset[:, -1]
# print input_dataset

# cross_validation(split)
train_input_dataset, test_input_dataset, train_output_classification, test_output_classification = train_test_split(
    input_dataset, output_classification, train_size=0.7, random_state=42)

# preprocess
input_dataset_normalized = preprocessing.normalize(input_dataset)
# print input_dataset_normalized
input_dataset_standardized = preprocessing.scale(input_dataset)
# print input_dataset_standardized

# model 1 for feature importamce
model_1 = ExtraTreesClassifier()
model_1.fit(input_dataset, output_classification)
# print model_1.feature_importances_

# model 2
model_2 = LogisticRegression()
rfe = RFE(model_2, 3)
rfe = rfe.fit(input_dataset, output_classification)
# print rfe.support_
# print rfe.ranking_

# model 3
'''
in general use training dataset to fit model
and use test dataset to make prediction
then compare its results with expected y
'''
model_3 = GaussianNB()
model_3.fit(train_input_dataset, train_output_classification)
excepted = test_output_classification
predicted = model_3.predict(test_input_dataset)
print metrics.classification_report(excepted, predicted)
print metrics.confusion_matrix(excepted, predicted)

# optimization 1
'''
this part is used to find the best params
'''
alphas = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
model_4 = Ridge()
grid = GridSearchCV(estimator=model_4, param_grid=dict(alpha=alphas))
grid.fit(input_dataset, output_classification)
# print grid.best_score_
# print grid.best_estimator_.alpha

# optimization 2
param_grid = {'alpha': sp_rand()}
rsearch = RandomizedSearchCV(estimator=model_4, param_distributions=param_grid, n_iter=100)
rsearch.fit(input_dataset, output_classification)
# print rsearch.best_score_
# print rsearch.best_estimator_.alpha
