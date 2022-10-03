import numpy as np
import math
k=64
for i in range(0,k):
        A1=np.load("./adj/A"+str(i)+".npy")
        distances = A1[~np.isinf(A1)].flatten()
        std = distances.std()
        if std < 0.00001:
            std = 1
        adj_mx = np.exp(-np.square(A1 / std))
        adj_mx[adj_mx < 0.1] = 0
        A1=adj_mx

        def cal_gcn_matrix(A):

            A=A.A
            I = np.diag(np.ones(A.shape[0], dtype=np.float32))
            D_diag = A.sum(axis=1)
            D_ = np.diag(np.power(D_diag, -1 ))
            D_[D_ == np.inf] = 0

            return 0.5*np.matmul(D_, A)

        A1= np.mat(A1)
        LA1=cal_gcn_matrix(A1)

        np.save("./LAdj/LA" + str(i), LA1)






