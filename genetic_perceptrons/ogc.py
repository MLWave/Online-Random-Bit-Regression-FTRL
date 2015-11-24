"""
	Online Genetic Classifier
	
	Totally experimental code/proof-of-concept.
"""
from __future__ import division
from collections import defaultdict
import sys
import random
from math import exp, log

class GeneticClassifier(object):
	def __init__(self, verbose=2, loc_python="python", get_train_data_function="", get_test_data_function="", loss_function="log_loss", custom_loss_function="", random_state=42):
		self.loc_python = loc_python
		self.verbose = verbose
		self.get_train_data_function = get_train_data_function
		self.get_test_data_function = get_test_data_function
		random.seed(random_state)
		if len(custom_loss_function) > 0:		
			self.loss = custom_loss_function
		else:
			if loss_function == "log_loss":
				self.loss = self.log_loss
			elif loss_function == "mse":
				self.loss = self.mse
			else:
				sys.exit("invalid loss function specified. Pick any of ['log_loss', 'mse']")
		self.minmax = defaultdict(lambda: defaultdict(float))
		
	def __repr__(self):
		return "GeneticClassifier()"
	
	def log_loss(self,y,p):
		p = max(min(p, 1. - 10e-15), 10e-15)
		return -log(p) if y == 1. else -log(1. - p)
		
	def mse(self,y_real,y_pred):
		print "ddd"
	
	def random_perceptron(self, size=3):
		perceptron = []
		for feature_index in random.sample(self.minmax.keys(),size):
			perceptron.append((random.uniform(-1,1), feature_index))
		return perceptron

	def calculate_perceptron(self, x, perceptron):
		#print sum([random_weight*x[feature_index] for random_weight, feature_index in perceptron[:-1]]), perceptron[-1], "drek"
		return sum([random_weight*x[feature_index][1] for random_weight, feature_index in perceptron[:-1]])
			
	def data_gen(self, data_generator):
		return data_generator
	
	def bounded_sigmoid(self, wTx):
		return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))	
	
	def fit(self, data_generator,data_generator2,data_generator3):
		# Calculate min and max for every column
		k = self.data_gen(data_generator)
		
		if self.verbose > 0:
			print("calculating min and max for every feature_index")
		for i, (x, y) in enumerate(k):
			for feature_index, feature_val in x:
				if i == 0:
					self.minmax[feature_index]["min"] = feature_val
					self.minmax[feature_index]["max"] = feature_val
				else:
					if feature_val < self.minmax[feature_index]["min"]:
						self.minmax[feature_index]["min"] = feature_val
					if feature_val > self.minmax[feature_index]["max"]:
						self.minmax[feature_index]["max"] = feature_val
			if self.verbose > 0:
				if i % 1000 == 0:
					print(i)
		#print self.minmax

		# generate n random perceptrons with random threshold between min,max.
		perceptrons = []
		for i in range(5000):
			perceptrons.append(self.random_perceptron())

		#k = data_generator	
		# calculate fitness of generation
		fitness = defaultdict(list)
		fitness = defaultdict(lambda: defaultdict(int))
		fitness = defaultdict(float)
		#k = self.data_gen(data_generator2)
		for i, (x, y) in enumerate(data_generator2):
			#print "kek"
			for perceptron_id, perceptron in enumerate(perceptrons):
				#print perceptron
				#print self.calculate_perceptron(x, perceptron)
				#fitness[perceptron_id].append(self.calculate_perceptron(x, perceptron))
				#print self.bounded_sigmoid(self.calculate_perceptron(x, perceptron)), perceptron
				fitness[perceptron_id] += self.log_loss( y, self.bounded_sigmoid(self.calculate_perceptron(x, perceptron)) )
		#print fitness
		"""
		fitness_keys = fitness.keys()
		for k in fitness_keys:
			#print k, fitness[k], perceptrons[k]
			total = sum(fitness[k].values())
			for label in fitness[k]:
				fitness[k][label] = fitness[k][label] / total
		for k in fitness_keys:
			print k, fitness[k]
		"""
		fittest = []
		for f in sorted(fitness, key=fitness.get)[:3]:
			#print f, fitness[f], fitness[f] / i, perceptrons[f]
			fittest.append(perceptrons[f])
		kk = []	
		for i, (x, y) in enumerate(data_generator3):
			pred = []
			for perceptron_id, perceptron in enumerate(fittest):
				pred.append(self.bounded_sigmoid(self.calculate_perceptron(x, perceptron)))
			#print y, sum(pred) / len(pred)
			kk.append((sum(pred) / len(pred), y))
		from sklearn.metrics import roc_auc_score
		preds = []
		y_real = []
		for k in sorted(kk):
			preds.append(k[0])
			y_real.append(k[1])
			print k
		print roc_auc_score(y_real, preds)
		print fittest	
clf = GeneticClassifier()

from sklearn.datasets import load_boston, load_digits
X, ys = load_boston().data, load_boston().target
X, ys = load_digits().data, load_digits().target

X_bin = []
y_bin = []
for x, y in zip(X,ys):
	if y == 1 or y == 0:
		X_bin.append(list(x))
		y_bin.append(int(y))

def get_data(X,ys):
	for x, y in zip(X, ys):
		yield [(e,f) for e,f in enumerate(x)], y

clf.fit(get_data(X_bin[:200],y_bin[:200]),get_data(X_bin[:200],y_bin[:200]),get_data(X_bin[200:],y_bin[200:]))