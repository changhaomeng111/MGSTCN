import numpy as np
import math
k=64
P=np.zeros((325, 325))
LAlengthlist=[]
for i in range(0,k):
    A1=np.load("./LA"+str(i)+".npy")
    LAlengthlist.append(A1.shape[0])

for i in range(0,k-1):
        LAlengthlist[i+1]+= LAlengthlist[i]

for i in range(0,k):
        A1=np.load("./LA"+str(i)+".npy")
        shape = A1.shape
        for x in range(0, shape[0]):
            for y in range(0, shape[1]):
                if i==0:
                    P[x,y] = A1[x, y]
                else:
                    P[(LAlengthlist[i-1])+x,(LAlengthlist[i-1])+y]=A1[x, y]

for i in range(0,k):
    for j in range(0,k):
        if i != j:
            A1 = np.load("./LA" + str(i) + "-LA" + str(j) + ".npy")
            shape = A1.shape
            for x in range(0, shape[0]):
                for y in range(0, shape[1]):
                    if i == 0:
                        P[x, (LAlengthlist[j - 1])+ y] = A1[x, y]
                    elif j==0:
                        P[(LAlengthlist[i - 1])+x, y] = A1[x, y]
                    else:
                        P[(LAlengthlist[i - 1]) + x, (LAlengthlist[j - 1]) + y] = A1[x, y]
P = np.mat(P)
def cal_gcn_matrix1(A):

    A=A.A
    D_diag = A.sum(axis=1)
    Dz_ = np.diag(np.power(D_diag, 1 / 2))
    Df_ = np.diag(np.power(D_diag, -1 / 2))
    Dz_[Dz_ == np.inf] = 0
    Df_[Df_ == np.inf] = 0
    return np.matmul(np.matmul(Dz_, A), Df_)

def cal_gcn_matrix2_zhuanzhi(A):

    A=A.A.T.conjugate()
    D_diag = A.sum(axis=1)
    Dz_ = np.diag(np.power(D_diag, 1 / 2))
    Df_ = np.diag(np.power(D_diag, -1 / 2))
    Dz_[Dz_ == np.inf] = 0
    Df_[Df_ == np.inf] = 0
    return np.matmul(np.matmul(Df_, A), Dz_)

I = np.diag(np.ones(P.shape[0], dtype=np.float32))
LP=I-(cal_gcn_matrix1(P)+cal_gcn_matrix2_zhuanzhi(P))/2
np.save("LP"+str(k),LP)