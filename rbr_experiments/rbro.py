# Coding up the algorithm from http://arxiv.org/abs/1501.02990 
# "Random Bits Regression: a Strong General Predictor for Big Data"

import random
from datetime import datetime

def create_var_subset(x,size=3):
	# (1) Randomly select a small subset of variables, e.g. x1, x3, x6.
	return random.sample([i for i in range(len(x))],min(size,len(x)))
	
def assign_weights(var_subset):
	# (2) Randomly assign weights to each selected variables. The weights 
	# are sampled from standard normal distribution, for example, 
	# w1, w3, w6~N(0,1)
	return [(random.random(),i) for i in var_subset]
	
def obtain_weighted_sum(x, weighted_var_subset):
	# (3) Obtain the weighted sum for each sample, for example
	# (w1*x1) + (w3*x3) + (w6*x6) = zi for the ith sample.
	weighted_sum = 0
	for w, i in weighted_var_subset:
		weighted_sum += w * x[i]
	return weighted_sum
	
def pick_random_threshold(weighted_sums):
	# (4) Randomly pick one zi from the n generated as the threshold T.
	return random.choice(weighted_sums)
	
def assign_bit(weighted_sum, threshold):
	# (5) Assign bits values to fk according to the threshold T
	# If zi >= T then 1 else 0
	if weighted_sum >= threshold:
		return 1
	else:
		return 0

def process(data, K=100, size=3):
	# The process is repeated K times.
	start = datetime.now()
	data_bits = []
	for k in range(K):
		var_subset = create_var_subset(data[0],size=size) # 1
		weighted_var_subset = assign_weights(var_subset) #2
		weighted_sums = []
		for x in data:
			weighted_sums.append(obtain_weighted_sum(x, weighted_var_subset)) # 3
			# The first feature is fixed to 1 to act as the interceptor. 
			if k == 0:
				data_bits.append([1])

		random_threshold = pick_random_threshold(weighted_sums) # 4 (Try picking multiple thresholds or entropy)

		for i, (x, data_bit) in enumerate(zip(data, data_bits)):
			data_bit.append( assign_bit(obtain_weighted_sum(x, weighted_var_subset),random_threshold) ) # 5

		if k % 1000 == 0:
			print k, datetime.now() - start
	return data_bits

random.seed(100)

from sklearn import datasets
data, y = datasets.load_digits().data, datasets.load_digits().target
data = [list(x) for x in data]

data_bits = process(data, 10000, 3) # We generate ~10^4-10^6 random binary intermediate features for each sample.
	
from sklearn import linear_model, ensemble, svm, neighbors, cross_validation
import numpy as np

# Select predictive intermediate features by regularized linear/logistic regression.

# KNN Classifier without intermediate features
start = datetime.now()
clf = neighbors.KNeighborsClassifier()
scores = cross_validation.cross_val_score(clf, data, y,cv=20)
print clf, np.array(data).shape
print scores
print scores.mean()
print datetime.now() - start
print

# KNN Classifier with intermediate features
start = datetime.now()
scores = cross_validation.cross_val_score(clf, data_bits, y, cv=20)
print clf, np.array(data_bits).shape
print scores
print scores.mean()
print datetime.now() - start
print

# SGD Classifier without intermediate features
start = datetime.now()
clf = linear_model.SGDClassifier(loss="log", penalty="l2", n_iter=20, random_state=1, n_jobs=-1)
scores = cross_validation.cross_val_score(clf, data, y,cv=20)
print clf, np.array(data).shape
print scores
print scores.mean()
print datetime.now() - start
print

# SGD Classifier with intermediate features
start = datetime.now()
scores = cross_validation.cross_val_score(clf, data_bits, y, cv=20)
print clf, np.array(data_bits).shape
print scores
print scores.mean()
print datetime.now() - start
print

# Logistic Regression without intermediate features
start = datetime.now()
clf = linear_model.LogisticRegression()
scores = cross_validation.cross_val_score(clf, data, y,cv=20)
print clf, np.array(data).shape
print scores
print scores.mean()
print datetime.now() - start
print

# Logistic Regression with intermediate features
start = datetime.now()
scores = cross_validation.cross_val_score(clf, data_bits, y, cv=20)
print clf, np.array(data_bits).shape
print scores
print scores.mean()
print datetime.now() - start
print

# Standard RF without features
start = datetime.now()
clf = ensemble.ExtraTreesClassifier(n_estimators=500,random_state=1,n_jobs=-1)
scores = cross_validation.cross_val_score(clf, data, y, cv=20)
print clf, np.array(data).shape
print scores
print scores.mean()
print datetime.now() - start
print

start = datetime.now()
clf = ensemble.ExtraTreesClassifier(n_estimators=500,random_state=1,n_jobs=-1)
scores = cross_validation.cross_val_score(clf, data_bits, y, cv=20)
print clf, np.array(data_bits).shape
print scores
print scores.mean()
print datetime.now() - start
print

start = datetime.now()
clf = ensemble.RandomForestClassifier(n_estimators=500,n_jobs=-1,random_state=1)
scores = cross_validation.cross_val_score(clf, data, y, cv=20)
print clf, np.array(data).shape
print scores
print scores.mean()
print datetime.now() - start
print

start = datetime.now()
clf = ensemble.RandomForestClassifier(n_estimators=500,n_jobs=-1,random_state=1)
scores = cross_validation.cross_val_score(clf, data_bits, y, cv=20)
print clf, np.array(data_bits).shape
print scores
print scores.mean()
print datetime.now() - start
print

start = datetime.now()
clf = svm.SVC(kernel="linear")
scores = cross_validation.cross_val_score(clf, data, y, cv=20)
print clf, np.array(data).shape
print scores
print scores.mean()
print datetime.now() - start
print

start = datetime.now()
clf = svm.SVC(kernel="linear")
scores = cross_validation.cross_val_score(clf, data_bits, y, cv=20)
print clf, np.array(data_bits).shape
print scores
print scores.mean()
print datetime.now() - start
print