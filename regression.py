## Code by Huankang Guan        ##  
## huankguan2-c@my.cityu.edu.hk ##
##------------------------------##
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

class LogReg:
    def __init__(self, c=1000.0):
        self.c = c
        pass
    
    def train(self, data, labels):
        # self.clf = LogisticRegression(C=self.c, penalty='l2', tol=0.05, solver='saga',max_iter=10000)
        self.clf = LogisticRegression(multi_class='ovr', max_iter=10000, C=self.c)
        # self.clf.fit( preprocessing.scale(data), labels )
        self.clf.fit( data, labels )
    
    def predicts(self, xs):
        return self.clf.predict(xs)

class RLSRwMultiClass: ## onehot
    def __init__(self, n_classes=10, lam=100.0):
        self.n_classes = n_classes
        self.lam = lam
    
    def phi(self, xs):
        n, d = xs.shape
        bias = np.ones( (n,1), dtype=float )
        return np.concatenate( (xs, bias), axis=1 )
    
    def extend(self, y):
        n = y.shape[0]
        ret = np.zeros( (n, self.n_classes), dtype=float)
        for i in range(n):
            ret[i][ int(y[i]) ] = 1.0
        return ret
    
    def train(self, data, labels):
        data = self.phi(data)
        n, d = data.shape
        Y = self.extend(labels)
        pphi = np.dot(data.transpose(), data) + np.eye(d)*self.lam
        self.w = np.dot(np.dot(np.linalg.inv(pphi), data.transpose()), Y)
        ## the shape of w=dxc, d = n_features+1

    def predict(self, x):
        ans = np.dot( self.w.transpose(), self.phi(x.reshape(1,-1)).transpose() )
        return np.argmax(ans)

    def predicts(self, xs):
        ans = np.dot( self.phi(xs), self.w )
        return np.argmax( ans, axis=1)


class RLSovr: ## one-vs-rest (one-vs-all)
    def __init__(self, lam=100.0):
        self.lam = lam
    
    def phi(self, xs):
        n, d = xs.shape
        bias = np.ones( (n,1), dtype=float )
        return np.concatenate( (xs, bias), axis=1 )

    def ovrtrain(self, data, labels):
        n, d = data.shape
        pphi = np.dot( data.transpose(), data )+self.lam*np.eye(d)
        return np.dot(np.dot(np.linalg.inv(pphi), data.transpose()), labels.reshape(-1,1))

    def train(self, data, labels):
        data = self.phi(data)
        n, d = data.shape
        
        self.w = {}
        unilabel = np.unique(labels)
        self.label = unilabel
        for _ in unilabel:
            L = np.array( [ {True:1.0, False:-1.0}[labels[i]==_] for i in range(n)] )
            self.w[_] = self.ovrtrain(data, L)
        
    def predict(self, x):
        for ind in self.label:
            w = self.w[ind].reshape(-1)
            px = self.phi( x.reshape(1,-1) ).reshape(-1)
            if np.dot(w,px)>0.0:return ind
        return self.label[-1]

    def predict2(self, x):
        candidate = -1
        val = -1
        for ind in self.label:
            w = self.w[ind].reshape(-1)
            px = self.phi( x.reshape(1,-1) ).reshape(-1)
            v = np.dot(w,px)
            if val==-1 or v>val:
                val = v
                candidate = ind
        return candidate

    def predicts(self, xs):
        # n, d = xs.shape
        # ans = [self.predict(xs[_]) for _ in range(n)]
        # return np.array(ans)
        n, d = xs.shape
        ans = [self.predict2(xs[_]) for _ in range(n)]
        return np.array(ans)

