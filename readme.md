# Randomly Follow the Regularized Leader

This is a class containing a binary classifier for online machine learning. It employs approaches from the papers:

rftrl.RandomLeaderClassifier(alpha=0.1, beta=1., l1=0., l2=1., nr_projections=10000, max_projections=0, 
					subsample_projections=1., size_projections=3, random_state=0,
					verbose=0):
					
## Parameters

alpha. Learning Rate.
beta. Smoothing parameter for adaptive learning rate
l1. L1 Regularization
l2. L2 Regularization
nr_projections. Number of random linear projections to create
max_projections. Not implemented.
subsample_projections. Float. Uses subsampling when making a first pass to create the random thresholds. This is more memory friendly for larger datasets.
size_projections. Number of (feature_value * random_weight) to use in the random linear functions.
random_state. Seed for replication.
Verbose. Int. Verbosity of classifier. Default=0.

## Usage

  clf = rftrl.RandomLeaderClassifier(nr_projections=500, random_state=1, size_projections=3, verbose=1)
  
  # Project data
  clf.project(X_train)
  
  # Train
  loss = 0
  for e, (x,y) in enumerate(zip(X_train,y)):
	clf.fit(x,e,y)
	pred = clf.predict()
	loss += clf.logloss()
	clf.update(pred)
  
  # Test
  y = 1
  for e, x in enumerate(X_test):
    clf.fit(x,e,y)
    pred = clf.predict()
    print("%s,%s"%(e,pred))

## References

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

  Tinrtgu's Beat the Benchmark online FTRL proximal script's
  Beat the benchmark with less then 200MB of memory.
https://www.kaggle.com/c/criteo-display-ad-challenge/forums/t/10322/beat-the-benchmark-with-less-then-200mb-of-memory/53737
https://www.kaggle.com/c/tradeshift-text-classification/forums/t/10537/beat-the-benchmark-with-less-than-400mb-of-memory/