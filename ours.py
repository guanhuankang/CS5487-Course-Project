## Code by Huankang Guan        ##  
## huankguan2-c@my.cityu.edu.hk ##
##------------------------------##
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


import util

class LogReg:
    def __init__(self):
        pass
    
    def train(self, data, labels):
        self.clf = LogisticRegression(multi_class='ovr', max_iter=10000, C=0.01)
        self.clf.fit( data, labels )
    
    def predicts(self, xs):
        return self.clf.predict(xs)


class OURS:
    def __init__(self):
        self.pair = {
            0:0,
            6:0,
            1:1,
            7:1,
            2:2,
            8:2,
            3:3,
            5:3,
            4:4,
            9:4,
        }

    def ext(self, labels, i):
        n = labels.shape[0]
        ret = np.zeros(n)
        xid = {0:6,1:7,2:8,3:5,4:9}
        ret[ np.where(labels==xid[int(i)]) ] = 1.0
        return ret

    def reext(self, pred, i):
        n = pred.shape[0]
        ret = np.ones(n)
        xid = {0:6,1:7,2:8,3:5,4:9}
        bigind = np.where(pred==1)[0]
        smind = np.where(pred==0)[0]
        ret[smind] = i
        ret[bigind] = xid[i]
        return ret

    def train(self, data, labels):
        pair_gt = labels.copy()
        n, d = data.shape
        for i in range(n):
            pair_gt[i] = self.pair[int(pair_gt[i])]
        
        self.pair_clf = LogReg()
        self.pair_clf.train(data, pair_gt)

        pred = self.pair_clf.predicts(data)
        self.clfs = {}
        for i in range(5):
            indices = np.where( pred==i )[0]
            if indices.shape[0]<=0:
                self.clfs[i] = None; raise; continue
            self.clfs[i] = LogReg()
            self.clfs[i].train( data[indices], self.ext(labels[indices], i) )
    
    def predicts(self, xs):
        n, d = xs.shape
        pred = self.pair_clf.predicts(xs)
        results = np.zeros( n )
        for i in range(5):
            indices = np.where( pred==i )[0]
            if indices.shape[0]<=0:
                continue
            ret = self.clfs[i].predicts( xs[indices] )
            ret = self.reext(ret, i)
            results[indices] = ret
        return results
