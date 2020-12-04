## Code by Huankang Guan        ##  
## huankguan2-c@my.cityu.edu.hk ##
##------------------------------##
import numpy as np

class BayesClassifier:
    def __init__(self, K=10):
        self.K = K

    def train(self, data, labels):
        n, d = data.shape
        unilabel = np.unique(labels)
        self.u = {}
        self.sigma = {}
        self.labels = []
        self.sigma_inv = {}
        self.sigma_det = {}
        self.ratio = {}

        for _ in unilabel:
            self.labels.append(_)
            indexs = np.where(labels==_)[0]
            self.ratio[_] = len(indexs)*1.0 / n

            u = data[indexs].mean(axis=0)
            self.u[_] = u

            sigma = np.zeros( (d,d), dtype=float )
            for i in indexs:
                sigma += np.outer(data[i]-u, data[i]-u)
            sigma = sigma / indexs.shape[0]

            self.sigma[_] = sigma
            det = np.linalg.det(sigma)
            self.sigma_det[_] = det
            if det==0.0:
                self.sigma_inv[_] = np.linalg.inv(sigma+1e-7*np.eye(d))
                self.sigma_det[_] = 1e-9
            else:
                self.sigma_inv[_] = np.linalg.inv(sigma)
    
        return len(self.labels)
    
    def logGaussian(self, x, label):
        v = -np.log(self.sigma_det[label]) \
            -np.dot( x-self.u[label], np.dot(self.sigma_inv[label], x-self.u[label]) ) \
                + np.log( self.ratio[label] )
        return v

    def predict(self, x):
        ar = np.array([ self.logGaussian(x, _) for _ in self.labels])
        return self.labels[ np.argmax( ar ) ]
    
    def predicts(self, testSet):
        ans = []
        n, d = testSet.shape
        for i in range(n):
            ans.append(self.predict(testSet[i]))
        return np.array(ans)
