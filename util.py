## Code by Huankang Guan        ##  
## huankguan2-c@my.cityu.edu.hk ##
##------------------------------##
import numpy as np
from PIL import Image
import os

def accuracy(x, gt):
    y = x.reshape(-1)
    G = gt.reshape(-1)
    if y.shape!=G.shape:
        print("Not match! accuracy");raise
    unilabel = np.unique(gt)
    match = (y==G)*1.0
    accu = match.sum() / G.shape[0] * 100.0
    pcer = {}
    for _ in unilabel:
        indices = np.where(gt==_)[0]
        pcer[str(int(_))] = 100.0 - 100.0*match[indices].sum()/indices.shape[0]
    return accu, pcer

    # print(accu, pcer)
    # n = x.shape[0]
    # err = np.zeros( (10,10), dtype=float )
    # for i in range(n):
    #     c, r = int(x[i]), int(gt[i])
    #     err[r,c] += 1
    # err = err.tolist()
    # print([" "]+[str(_) for _ in range(10)])
    # cnt = 0
    # for _ in err:
    #     print(str(cnt)+" "+str(_)+"\n")
    #     cnt += 1
    # exit(0)
    
    # return accu, pcer

## trial = 1 or 2
def read(normalize=False, shuffle=False):
    data = {}
    path = "digits4000_txt/digits4000_txt"
    with open(path+"/digits4000_digits_labels.txt", "r") as f:
        label = []
        for line in f.readlines():
            label.append( np.array( list(map(float, line.split())), dtype="float") )
        gt = np.vstack(label).reshape(-1)
    
    with open(path+"/digits4000_digits_vec.txt", "r") as f:
        label = []
        for line in f.readlines():
            label.append( np.array( list(map(float, line.split())), dtype="float") )
        vec = np.vstack(label)
        
    if normalize:
        vec = vec.astype("float")/255.0
    if shuffle:
        np.random.seed(5487)
        np.random.shuffle(vec)
        np.random.shuffle(gt)

    return vec, gt

def pca(data, k=2):
    # pca = PCA(n_components=k)
    # pca.fit(data)
    # return pca.transform(data)

    n, d = data.shape
    u = data.mean(axis=0)
    sigma = np.zeros( (d,d), dtype=float )
    for i in range(n):
        sigma += np.outer(data[i]-u, data[i]-u)
    sigma = sigma / n    
    val, vec = np.linalg.eig(sigma)

    indexs = np.argsort(np.real(val))
    veck = vec[..., indexs[-k::]]
    return np.matmul(data-u, np.real(veck))

def display(arr):
    if np.max(arr)<2.0: arr = arr*255.0
    img = arr.reshape( (28,28) ).transpose()
    img = Image.fromarray( img.astype(np.uint8) )
    img.show()


def readChallenges(normalize=False, shuffle=False):
    path = "digits4000_txt/challenges"
    with open(path+"/cdigits_digits_labels.txt", "r") as f:
        label = []
        for line in f.readlines():
            label.append( np.array( list(map(float, line.split())), dtype="float") )
        gt = np.vstack(label).reshape(-1)
    
    with open(path+"/cdigits_digits_vec.txt", "r") as f:
        label = []
        for line in f.readlines():
            label.append( np.array( list(map(float, line.split())), dtype="float") )
        vec = np.vstack(label)
        
    if normalize:
        vec = vec.astype("float")/255.0
    if shuffle:
        np.random.seed(5487)
        np.random.shuffle(vec)
        np.random.shuffle(gt)

    return vec, gt

# data = read()
# print(data["train_gt"].shape)
# for i in range(20):
#     display(data["train_vec"][int(i*99)])