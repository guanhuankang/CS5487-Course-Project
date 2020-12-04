import matplotlib.pyplot as plt
import numpy as np

import util
from runners import Runner

from Bayesian import BayesClassifier
from Kmean import KMean
from KNN import KNN
from SVM import linearSVM, rbfSVM, polySVM
from LDA import LDA
from regression import LogReg, RLSRwMultiClass, RLSovr
from ours import OURS
from ourKNN import OURSKNN

runner = Runner()
names = ["Bayesian", "KMean", "KNN", "LDA", "SVM-linear", "SVM-RBF", "SVM-Poly", "LogReg", "RLSovr", "RLSonehot", "ours", "oursKNN"]
clfs = [BayesClassifier(), KMean(), KNN(), LDA(), linearSVM(), rbfSVM(), polySVM(), LogReg(), RLSovr(), RLSRwMultiClass(), OURS(), OURSKNN()]

names = ["ours", "oursKNN"]
clfs = [OURS(), OURSKNN()]

n_clf = len(names)

for i in range(n_clf):
    name, clf = names[i], clfs[i]
    log = name+"\n"

    print("name:",name)
    ret = runner.run(clf=clf, mode="test")
    print(ret); log += str(ret)+"\n"
    ret = runner.run(clf=clf, mode="PCA", n_component=200)
    print(ret); log += str(ret)+"\n"
    ret = runner.run(clf=clf, mode="LDA", n_component=9)
    print(ret); log += str(ret)+"\n"

    with open("result.log", "a+") as f:f.write(log)
    print("-----------------\n")

## challenges dataset
for i in range(n_clf):
    name, clf  = names[i], clfs[i]

    vec, gt = util.read(normalize=True, shuffle=False)
    clf.train(vec, gt)
    vec, gt = util.readChallenges(normalize=True)
    accu, pcer = util.accuracy( clf.predicts(vec), gt)
    log = "%s-accu:%f\n"%(name, accu)
    print( "%s-accu:%f\n"%(name, accu) )
    
    with open("result-challenges.log", "a+") as f:f.write(log)
    print("-----------------\n")
