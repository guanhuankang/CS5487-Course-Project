## Code by Huankang Guan        ##  
## huankguan2-c@my.cityu.edu.hk ##
##------------------------------##
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LDA:
    def __init__(self):
        pass

    def lda(self, data, labels, X, k=9):
        ## Note: k<=min(n_classes-1, n_features)
        self.reduction = LinearDiscriminantAnalysis(n_components=k)
        self.reduction.fit(data, labels)
        return self.reduction.transform(X)
    
    def train(self, data, labels):
        self.clf = LinearDiscriminantAnalysis()
        self.clf.fit(data, labels)
    
    def predict(self, x):
        return self.clf.predict( [x] )[0]

    def predicts(self, xs):
        return self.clf.predict(xs)