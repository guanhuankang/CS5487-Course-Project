## Code by Huankang Guan        ##  
## huankguan2-c@my.cityu.edu.hk ##
##------------------------------##
import numpy as np
from sklearn import svm

class linearSVM():
    def __init__(self):
        pass
    
    def train(self, data, labels):
        self.clf = svm.SVC(kernel="linear", decision_function_shape='ovr')
        self.clf.fit( data, labels )

    def predict(self, x):
        return self.clf.predict(x.reshape(1,-1))[0]

    def predicts(self, xs):
        ans = self.clf.predict( xs )
        return ans


class rbfSVM():
    def __init__(self):
        pass
    
    def train(self, data, labels):
        self.clf = svm.SVC(kernel="rbf", decision_function_shape='ovr')
        self.clf.fit( data, labels )

    def predict(self, x):
        return self.clf.predict(x.reshape(1,-1))[0]

    def predicts(self, xs):
        ans = self.clf.predict( xs )
        return ans


class polySVM():
    def __init__(self):
        pass
    
    def train(self, data, labels):
        self.clf = svm.SVC(kernel="poly", degree=3 , decision_function_shape='ovr')
        self.clf.fit( data, labels )

    def predict(self, x):
        return self.clf.predict(x.reshape(1,-1))[0]

    def predicts(self, xs):
        ans = self.clf.predict( xs )
        return ans

