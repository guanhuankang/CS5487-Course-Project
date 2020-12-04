import numpy as np
import heapq

class KNN:
    def __init__(self, K = 1):
        self.K = K

    def train(self, data, labels):
        self.data = data.copy()
        self.labels = labels.copy()

    def predict(self, x):
        pq = []
        n, d = self.data.shape

        for i in range(n):
            dis = np.sum( np.square( x - self.data[i] ) )
            if len(pq)<self.K:
                heapq.heappush( pq, (-dis, self.labels[i]) )
            else:
                if -pq[0][0] > dis:
                    heapq.heapreplace( pq, (-dis, self.labels[i]) )
        
        dic = {}
        for _ in pq:
            dis, ind = -_[0], _[1]
            if ind not in dic:
                dic[ind] = [1, dis]
            else:
                dic[ind][0] += 1
                dic[ind][1] += dis

        ## d/n smaller is better
        pq = []
        for k in dic:
            # pq.append( (1/dic[k][0], k) ) ## 1/n
            # pq.append( (dic[k][1]/dic[k][0], k) ) ## avg(d) = sum(d)/(n^1) 
            pq.append( (dic[k][1]/(dic[k][0]**2), k) ) ## avg(d)/n = sum(d)/(n^2) 
            # pq.append( (dic[k][1]/(dic[k][0]**3), k) ) ## sum(d)/(n^3) 
            # pq.append( (dic[k][1]/(dic[k][0]**4), k) ) ##  sum(d)/(n^4) 
            # pq.append( (dic[k][1]/(dic[k][0]**5), k) ) ## sum(d)/(n^5) 
        heapq.heapify( pq )
        return pq[0][1]

    def predicts(self, xs):
        n, d = xs.shape
        ans = [ self.predict(xs[_]) for _ in range(n) ]
        return np.array( ans )