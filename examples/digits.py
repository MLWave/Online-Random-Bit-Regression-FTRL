"""
  Author: Triskelion, HJ van Veen, info@mlwave.com
  
  Description:  
  
  Creates 3 classifiers. 
  Experiments with ensembling their predictions, and studies variance.
  Uses digits dataset (the "0"'s and "1"'s)
				
  Seems that: 
  
  One 'overfitted' classifier can improve the ensemble.
  Random_state change shows more difference than Random Forest.
  Averaging 3 classifiers with different random state increases robustness.
  We can approach SVM accuracy.
  Weighing the predictions by the 3 classifier's progressive validation loss can be better than unweighted average.
  Very similar to Vowpal Wabbit's -q and --cubic.
  
"""
from sklearn.datasets import load_digits
import rftrl

def logloss(act,pred):
  predicted = max(min(pred, 1. - 10e-15), 10e-15)
  return -log(predicted) if act == 1. else -log(1. - predicted)
    
if __name__ == "__main__":      
  X_train, y = load_digits().data, load_digits().target
        
  clf = rftrl.RandomLeaderClassifier(nr_projections=500, random_state=36, l2=1., size_projections=1, verbose=1)
  clf2 = rftrl.RandomLeaderClassifier(nr_projections=100000, random_state=37, l2=1., size_projections=3, verbose=1)
  clf3 = rftrl.RandomLeaderClassifier(nr_projections=1000, random_state=38, l2=1., size_projections=2, verbose=1)

  clf.project(X_train)
  clf2.project(X_train)
  clf3.project(X_train)

  loss = 0
  loss2 = 0
  loss3 = 0
  loss_ensemble = 0
  loss_ensemble_ranked = 0
  count = 0
  for e, (x,y) in enumerate(zip(X_train,y)):
    if y == 0 or y == 1: # make a binary problem
      count += 1.
      
      clf.fit(x,e,y)
      pred = clf.predict()
      loss += clf.logloss()
      clf.update(pred)
      
      clf2.fit(x,e,y)
      pred2 = clf2.predict()
      loss2 += clf2.logloss()
      clf2.update(pred2)
      
      clf3.fit(x,e,y)
      pred3 = clf3.predict()
      loss3 += clf3.logloss()
      clf3.update(pred3)
      
      leaders = sorted([(loss/count,pred), (loss2/count,pred2),  (loss3/count,pred3)])
      loss_ensemble_ranked += logloss(y,((leaders[0][1]*3)+(leaders[1][1]*2)+(leaders[2][1]*1))/6.)
      loss_ensemble += logloss(y,(pred+pred2+pred3)/3.)

      print("%f\t%s\t%f\t%f\t%f\t\t%f\t%f"%(pred, y, loss/count, loss2/count,  loss3/count, loss_ensemble/count, loss_ensemble_ranked/count))