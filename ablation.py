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

runner = Runner()

ret = runner.run(clf=KNN(K=20), mode="test")
print(ret)
exit(0)

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
