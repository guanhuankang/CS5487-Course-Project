## K-fold trial ##
import numpy as np
import util
from LDA import LDA

class Runner:
    def __init__(self, k=10):
        self.k = int(k)
        # path = "digits4000_txt/digits4000_txt"
    
    def divide(self, xid, train):
        n = train.shape[0]
        tr = []
        val = []
        for _ in range(n):
            if _%self.k==xid:
                val.append(_)
            else:
                tr.append(_)
        return np.array(tr, dtype=int), np.array(val, dtype=int)

    ## mode = train, test, PCA, LDA
    ## both PCA and LDA have no val dataset
    def run(self, clf, mode="train", n_component=9):
        vec, gt= util.read(normalize=True, shuffle=False)

        if mode=="train":
            ans = []
            for trial in ((0, 2000),(2000,0)):
                train = np.arange(2000) + trial[0]
                # test = np.arange(2000) + trial[1]
                valAccu = 0.0
                for _ in range(self.k):
                    tr, val = self.divide(_, train)
                    X = vec[tr]; V = vec[val]
                    Y = gt[tr]; U = gt[val]
                    clf.train(X, Y)
                    accu, pcer = util.accuracy(clf.predicts(V), U)
                    print("k-fold-%d-out of %d, accu:%f"%(_,self.k,accu))
                    valAccu += accu
                valAccu = valAccu / self.k
                ans.append(valAccu)
                # clf.train(vec[train], gt[train])
                # testAccu, pcer = util.accuracy( clf.predicts(vec[test]), gt[test] )
            print("AvgVal accu:", ans)
            log = "mode=train: %f, %f"%(ans[0], ans[1])
            return sum(ans)/2.0, log

        elif mode=="test":
            ans = []
            for trial in ((0, 2000),(2000,0)):
                train = np.arange(2000) + trial[0]
                test = np.arange(2000) + trial[1]
                clf.train(vec[train], gt[train])
                testAccu, pcer = util.accuracy( clf.predicts(vec[test]), gt[test] )
                ans.append([testAccu, pcer])
            accu1, accu2 = ans[0][0], ans[1][0]
            pcer1, pcer2 = ans[0][1], ans[1][1]
            
            mean = (accu1+accu2)/2.0
            std = np.sqrt(((accu1-mean)**2+(accu2-mean)**2)/2.0)
            avgpcer = [ (_,(pcer1[_]+pcer2[_])/2.0) for _ in pcer1]
            # print("mode=test", accu1, accu2)
            log = "mode=test: %f, %f"%(accu1, accu2)
            return mean, std, avgpcer, log

        elif mode=="PCA":
            ans = []
            for trial in ((0, 2000),(2000,0)):
                train = np.arange(2000) + trial[0]
                test = np.arange(2000) + trial[1]
                ## PCA ##
                vec = util.pca( vec, k = n_component )
                #########
                clf.train(vec[train], gt[train])
                testAccu, pcer = util.accuracy( clf.predicts(vec[test]), gt[test] )
                ans.append([testAccu, pcer])
            accu1, accu2 = ans[0][0], ans[1][0]
            pcer1, pcer2 = ans[0][1], ans[1][1]
            
            mean = (accu1+accu2)/2.0
            std = np.sqrt(((accu1-mean)**2+(accu2-mean)**2)/2.0)
            avgpcer = [ (_,(pcer1[_]+pcer2[_])/2.0) for _ in pcer1]
            # print("mode=PCA", accu1, accu2)
            log = "mode=PCA: %f, %f"%(accu1, accu2)
            return mean, std, avgpcer, log

        elif mode=="LDA":
            ans = []
            for trial in ((0, 2000),(2000,0)):
                train = np.arange(2000) + trial[0]
                test = np.arange(2000) + trial[1]
                ## LDA-Train ##
                lda = LDA()
                X = lda.lda(vec[train], gt[train], X=vec[train])
                clf.train(X, gt[train])
                X = lda.lda(vec[train], gt[train], X=vec[test])
                testAccu, pcer = util.accuracy( clf.predicts(X), gt[test] )
                ans.append([testAccu, pcer])
            accu1, accu2 = ans[0][0], ans[1][0]
            pcer1, pcer2 = ans[0][1], ans[1][1]
            
            mean = (accu1+accu2)/2.0
            std = np.sqrt(((accu1-mean)**2+(accu2-mean)**2)/2.0)
            avgpcer = [ (_,(pcer1[_]+pcer2[_])/2.0) for _ in pcer1]
            # print("mode=LDA", accu1, accu2)
            log = "mode=LDA: %f, %f"%(accu1, accu2)
            return mean, std, avgpcer, log