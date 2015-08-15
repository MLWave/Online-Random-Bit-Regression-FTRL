""" Author: Triskelion, HJ van Veen, info@mlwave.com

	This class implements a binary classifier for online learning, which is based on descriptions in the papers:
	
	Random Bit Regression (RBR).
	  Random Bits Regression: a Strong General Predictor for Big Data
	  Yi Wang, Yi Li, Momiao Xiong, Li Jin
	  http://arxiv.org/abs/1501.02990
	
	Follow the Regularized Leader (FTRL)
	  Ad Click prediction: A view from the trenches. 
	  H. Brendan McMahan, Gary Holt, D. Sculley, Michael Young, Dietmar Ebner, Julian Grady, Lan Nie, Todd Phillips, 
	  Eugene Davydov, Daniel Golovin, Sharat Chikkerur, Dan Liu, Martin Wattenberg, Arnar Mar Hrafnkelsson, Tom Boulos, 
	  Jeremy Kubica.
	  https://research.google.com/pubs/archive/41159.pdf
	  
	Random Bit Regression
	
	RBR works well on dense tall datasets. The algorithm is most succinctly described in the paper:
	
	1. Randomly select a subset of variables, eg: f1, f2, f3
	2. Assign random weights uniformly drawn from between 0 and 1 for each variable in the subset. eg: w1 = 0.4532134
	3. Obtain the weighted sum (z). eg: z = (f1 * w1) + (f2 * w2) + (f3 * w3)
	4. Randomly pick one threshold (t_random) generated from all z's (Z). eg: t_random = 15.34245
	5. Vectorize samples with bits according to the formula: if z > t_random then 1 else 0.
	
	Basically we add the result of many random linear functions (perceptrons) as binarized features to a sample: Random Bit Vectorization.

	Follow the Regularized Leader
	
	We then use a logistic regression algorithm with L2 regularization to do conventional supervised learning on this bit representation.
	
	The online FTRL (oFTRL) code is credit to tinrtgu (https://www.kaggle.com/ggglhf) . This is a categorical classifier that was used for 
	"ad click prediction"-competitions on Kaggle. It used the hashing trick to one-hot encode all the features and supported both L1 and 
	L2 regularization. 
	
	Modifications
	
	RBR
	
	We modify (relax) step 4. from the Random Bit Regression Algorithm. We don't want to generate all the thresholds for the 
	entire dataset, simply to obtain a single random threshold. If we do all that, then we may as well pick thresholds so they 
	better divide the classes. A single pass over a dataset or batch is still needed to get a random threshold for every random 
	linear function. Heavy subsampling and a max Z-size ensures the generation of random thresholds without wasting too much time
	building the vectorizers. There are other paths to check out: completely random thresholds, prenormalizing or online normalization 
	of features, and "Don't do linear functions, but Euclidean distance to first n noise-distorted samples".
	
	oFTRL
	
	oFTRL was originally a purely categorical classifier. Through bit vectorizing the features with random linear functions it can now 
	handle features which were originally floats or numerical. Another benefit is the added boost for none-linearity in problems.
	
	As we always know the length of our binary representation, we do not need the hashing trick. We can simply sparse encode:
	
	"11101" becomes "1:1 2:1 3:1 5:1"
	
	We call this modified algorithm "Randomly Follow the Regularized Leader"
"""
import numpy as np
from math import sqrt, exp, log

class RandomLeaderClassifier(object):
	def __init__(self, alpha=0.1, beta=1., l1=0., l2=1., nr_projections=10000, max_projections=0, 
					subsample_projections=1., size_projections=3, random_state=0,
					verbose=0):
		self.z = [0.] * (nr_projections+1)
		self.n = [0.] * (nr_projections+1)
		self.nr_projections = nr_projections
		self.alpha = alpha
		self.beta = beta
		self.l1 = l1
		self.l2 = l2
		self.size_projections = size_projections
		self.subsample_projections = subsample_projections
		self.max_projections = max_projections
		self.random_state = random_state
		self.verbose = verbose 
		self.w = {}
		self.X = []
		self.y = 0.
		self.random_thresholds = []
		self.random_indexes = []
		self.random_weights = []
		self.Prediction = 0.
	
	def sgn(self, x):
		if x < 0:
			return -1  
		else:
			return 1

	def project(self, X_train):
		if self.verbose > 0:
			print("Creating %s random projections on train set shaped %s"%(self.nr_projections,str(X_train.shape)))
			print("Using random seed %s"%(self.random_state))
		np.random.seed(self.random_state)
		self.random_indexes = np.random.randint(0, high=X_train.shape[1], size=(self.nr_projections, self.size_projections))
		self.random_weights = np.random.rand(self.nr_projections,self.size_projections)
		for e, x in enumerate(X_train):
			if e == 0:
				thresholds = np.sum(x[self.random_indexes] * self.random_weights, axis=1).reshape((1,self.nr_projections))
			else:
				if np.random.random() < self.subsample_projections:
					
					thresholds = np.r_[thresholds, np.sum(x[self.random_indexes] * self.random_weights, axis=1).reshape((1,self.nr_projections))]
				if self.max_projections > 0 and thresholds.shape[0] >= self.max_projections:
					if self.verbose > 0:
						print("Halting.")
					break

		random_thresholds = []
		for column_id in range(self.nr_projections):
			random_thresholds.append(thresholds[np.random.randint(0,high=thresholds.shape[0])][column_id])
		self.random_thresholds = np.array(random_thresholds)

	
	def fit(self,x,sample_id,label):
		self.ID = sample_id
		self.y = float(label)
		
		thresholds = np.sum(x[self.random_indexes] * self.random_weights, axis=1).reshape((1,self.nr_projections))
		bools = thresholds > self.random_thresholds
		
		self.X = [e+1 for e, f in enumerate(list(bools.astype(int)[0])) if f == 1 ] # Sparse encoding the bitstring
		self.X = [0] + self.X # Prefix with a bias term

	def logloss(self):
		act = self.y
		pred = self.Prediction
		predicted = max(min(pred, 1. - 10e-15), 10e-15)
		return -log(predicted) if act == 1. else -log(1. - predicted)

	def predict(self):
		W_dot_x = 0.
		w = {}
		for i in self.X:
			if abs(self.z[i]) <= self.l1:
				w[i] = 0.
			else:
				w[i] = (self.sgn(self.z[i]) * self.l1 - self.z[i]) / (((self.beta + sqrt(self.n[i]))/self.alpha) + self.l2)
			W_dot_x += w[i]
		self.w = w
		self.Prediction = 1. / (1. + exp(-max(min(W_dot_x, 35.), -35.)))
		return self.Prediction

	def save(self,file_path):
		return False

	def load(self,file_path):
		return False
	
	def update(self, prediction): 
		for i in self.X:
			g = (prediction - self.y)
			sigma = (1./self.alpha) * (sqrt(self.n[i] + g*g) - sqrt(self.n[i]))
			self.z[i] += g - sigma*self.w[i]
			self.n[i] += g*g