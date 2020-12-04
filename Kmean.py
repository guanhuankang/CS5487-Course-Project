## Code by Huankang Guan        ##  
## huankguan2-c@my.cityu.edu.hk ##
##------------------------------##
import numpy as np

class KMean:
    def __init__(self):
        pass

    def train(self, data, labels):
        self.u = {}
        dic = {}
        num = {}
        n, d = data.shape
        for i in range(n):
            if labels[i] not in dic:
                dic[labels[i]] = data[i]
                num[labels[i]] = 1
            else:
                dic[labels[i]] += data[i]
                num[labels[i]] += 1
        for k in dic:
            self.u[k] = dic[k]*1.0/num[k]
        del dic
    
    def predict(self, x):
        dmin = -1
        ind = -1
        for k in self.u:
            dis = np.sum( np.square(x - self.u[k]) )
            # dis = np.sum( np.abs(x - self.u[k]) )
            if dmin==-1 or dmin>dis:
                dmin = dis
                ind = k
        return ind
    
    def predicts(self, xs):
        n, d = xs.shape
        ans = [ self.predict(xs[_]) for _ in range(n) ]
        return np.array(ans)
