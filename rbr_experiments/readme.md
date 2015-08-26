# Experiment offline RBR on digits
We write the algorithm to be as close to the paper as possible. Then we use a toy dataset `digits` shaped (1797, 64) with 10 classes. We look at algorithm performance of using 10^4 intermediate features of subset size 3.

## Results
20-fold CV acc. | Vectors | Algo
--- | --- | ---
0.981674170670 | **RBR** | **SVM**
0.981230593359 | RAW | KNN
0.978857040929 | RAW | ET
0.974951330371 | RBR | **LOGREG**
0.974470711080 | RBR | ET
0.972307524295 | RBR | KNN
0.971636906430 | RBR | **RF**
0.971165709003 | RAW | SVM
0.967125864925 | RAW | RF
0.965967668687 | RBR | **SGD**
0.946672851823 | RAW | LOGREG
0.916611431522 | RAW | SGD

## Prelim
RBR SVM took `0:04:24.457000` vs. RAW KNN `0:00:00.714000`. RBR improved SVM, Logreg, RF and SGD over using the RAW original features.

Logistic Regression took a long time with 10k RBR features. All-in-all RBR LOGREG could be a useful diverse addition to an ensemble.

## Console
```
0 0:00:00.012000
1000 0:00:08.505000
2000 0:00:17.046000
3000 0:00:25.596000
4000 0:00:34.179000
5000 0:00:42.764000
6000 0:00:51.321000
7000 0:00:59.911000
8000 0:01:08.479000
9000 0:01:17.042000

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_neighbors=5, p=2, weights='uniform') (1797L, 64L)
[ 0.92631579  0.97849462  0.98901099  1.          0.98888889  0.97777778
  0.98888889  0.96666667  0.98888889  0.94444444  1.          0.98888889
  0.98888889  1.          0.98876404  0.98876404  1.          0.95454545
  0.97701149  0.98837209]
0.981230593359
0:00:00.714000

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_neighbors=5, p=2, weights='uniform') (1797L, 10001L)
[ 0.91578947  0.97849462  0.98901099  0.98888889  0.98888889  0.97777778
  0.95555556  0.96666667  0.96666667  0.93333333  0.98888889  0.97777778
  0.97777778  0.98888889  1.          0.98876404  1.          0.90909091
  0.96551724  0.98837209]
0.972307524295
0:01:47.331000

SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='log', n_iter=20, n_jobs=-1,
       penalty='l2', power_t=0.5, random_state=1, shuffle=True, verbose=0,
       warm_start=False) (1797L, 64L)
[ 0.89473684  0.90322581  0.87912088  0.94444444  0.97777778  0.9
  0.88888889  0.91111111  0.87777778  0.9         0.97777778  0.98888889
  0.98888889  0.87777778  0.94382022  0.93258427  0.85393258  0.80681818
  0.91954023  0.96511628]
0.916611431522
0:00:03.182000

SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='log', n_iter=20, n_jobs=-1,
       penalty='l2', power_t=0.5, random_state=1, shuffle=True, verbose=0,
       warm_start=False) (1797L, 10001L)
[ 0.94736842  0.96774194  1.          0.98888889  0.98888889  0.94444444
  0.93333333  0.96666667  0.97777778  0.94444444  1.          0.97777778
  0.97777778  0.97777778  0.98876404  0.97752809  0.97752809  0.875
  0.94252874  0.96511628]
0.965967668687
0:01:13.966000

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr',
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0) (1797L, 64L)
[ 0.89473684  0.93548387  0.95604396  0.96666667  0.95555556  0.94444444
  0.92222222  0.93333333  0.92222222  0.97777778  0.98888889  0.98888889
  0.98888889  0.95555556  0.97752809  0.96629213  0.91011236  0.81818182
  0.96551724  0.96511628]
0.946672851823
0:00:06.033000

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr',
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0) (1797L, 10001L)
[ 0.94736842  0.96774194  1.          0.98888889  0.98888889  0.97777778
  0.94444444  0.97777778  0.97777778  0.96666667  0.98888889  0.98888889
  0.97777778  0.98888889  0.98876404  0.98876404  0.98876404  0.93181818
  0.95402299  0.96511628]
0.974951330371
0:09:41.141000

ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
           max_depth=None, max_features='auto', max_leaf_nodes=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
           oob_score=False, random_state=1, verbose=0, warm_start=False) (1797L, 64L)
[ 0.93684211  0.97849462  1.          1.          0.98888889  0.97777778
  0.97777778  0.96666667  0.97777778  0.97777778  0.98888889  0.98888889
  0.97777778  1.          0.98876404  0.98876404  1.          0.94318182
  0.97701149  0.94186047]
0.978857040929
0:00:16.637000

ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
           max_depth=None, max_features='auto', max_leaf_nodes=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
           oob_score=False, random_state=1, verbose=0, warm_start=False) (1797L, 10001L)
[ 0.91578947  0.96774194  1.          0.98888889  0.98888889  0.97777778
  0.96666667  0.96666667  0.96666667  0.98888889  1.          1.
  0.98888889  0.97777778  0.97752809  0.98876404  0.96629213  0.94318182
  0.96551724  0.95348837]
0.97447071108
0:02:19.255000

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
            oob_score=False, random_state=1, verbose=0, warm_start=False) (1797L, 64L)
[ 0.93684211  0.96774194  0.98901099  0.97777778  0.98888889  0.97777778
  0.95555556  0.95555556  0.96666667  0.96666667  0.98888889  0.98888889
  0.96666667  0.96666667  0.97752809  0.98876404  0.96629213  0.92045455
  0.95402299  0.94186047]
0.967125864925
0:00:18.448000

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
            oob_score=False, random_state=1, verbose=0, warm_start=False) (1797L, 10001L)
[ 0.92631579  0.95698925  1.          0.98888889  0.97777778  0.97777778
  0.96666667  0.96666667  0.97777778  0.98888889  0.98888889  0.98888889
  0.98888889  0.97777778  0.97752809  0.98876404  0.95505618  0.93181818
  0.96551724  0.94186047]
0.97163690643
0:01:42.010000

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
  kernel='linear', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False) (1797L, 64L)
[ 0.91578947  0.98924731  0.98901099  1.          0.98888889  0.96666667
  0.95555556  0.96666667  0.95555556  0.96666667  1.          0.97777778
  0.95555556  0.96666667  0.96629213  1.          0.98876404  0.92045455
  0.97701149  0.97674419]
0.971165709003
0:00:01.275000

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
  kernel='linear', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False) (1797L, 10001L)
[ 0.93684211  0.97849462  1.          1.          0.97777778  0.97777778
  0.97777778  1.          0.97777778  0.97777778  1.          1.
  0.98888889  0.98888889  1.          0.98876404  1.          0.92045455
  0.96551724  0.97674419]
0.98167417067
0:04:24.457000
```