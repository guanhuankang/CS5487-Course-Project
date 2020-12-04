## CS5487 Course Project

Group-5: Huankang Guan, Xianbin Zhu

Project: default course project

-------------------------------------------------------------------------

## Run

python main.py
## Result
The results will be save to result.log, in which you can find all the selected methods' accuracy and per-class-error-rate. For example,
```python
Bayesian
(89.4, 0.6499999999999986, [('0', 3.5), ('1', 2.25), ('2', 9.75), ('3', 14.25), ('4', 12.0), ('5', 23.0), ('6', 5.25), ('7', 11.25), ('8', 19.0), ('9', 5.75)], 'mode=test: 90.050000, 88.750000')
(64.325, 0.07500000000000284, [('0', 41.5), ('1', 16.25), ('2', 34.75), ('3', 60.25), ('4', 44.5), ('5', 41.25), ('6', 24.25), ('7', 27.75), ('8', 39.5), ('9', 26.75)], 'mode=PCA: 64.250000, 64.400000')
(79.55000000000001, 0.8500000000000014, [('0', 7.75), ('1', 7.75), ('2', 24.75), ('3', 27.0), ('4', 18.5), ('5', 27.0), ('6', 17.75), ('7', 22.25), ('8', 28.75), ('9', 23.0)], 'mode=LDA: 80.400000, 78.700000')
```

The first line is the method name (Bayesian). 

The second line is (mean accuracy, std, per-class-error-rate, mode(accuracy of the first trial and the second trial))

The third line is (mean accuracy (PCAas dimensionality reduction), std, per-class-error-rate, mode(accuracy of the first trial and the second trial))

The forth line is (mean accuracy (LDA as dimensionality reduction), std, per-class-error-rate, mode(accuracy of the first trial and the second trial))



For challenging dataset, you can find the accuracy of each method in file "result-challenges.log". 

## Thank you for reading

