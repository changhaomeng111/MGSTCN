import numpy as np
import os
import shutil


class Supra_Laplace(object):
    def __init__(self,Laplace_npy=""):
        self.Laplace_npy=Laplace_npy

def cal_gcn_matrix(A):
    A=A.A
    D_diag = A.sum(axis=1)
    Dz_ = np.diag(np.power(D_diag, 1 / 2))
    Df_ = np.diag(np.power(D_diag, -1 / 2))
    Dz_[Dz_ == np.inf] = 0
    Df_[Df_ == np.inf] = 0
    return np.matmul(np.matmul(Dz_, A), Df_)

def cal_gcn_matrix_T(A):
    A=A.A.T.conjugate()
    D_diag = A.sum(axis=1)
    Dz_ = np.diag(np.power(D_diag, 1 / 2))
    Df_ = np.diag(np.power(D_diag, -1 / 2))
    Dz_[Dz_ == np.inf] = 0
    Df_[Df_ == np.inf] = 0
    return np.matmul(np.matmul(Df_, A), Dz_)

def construct_supra_Laplace(prefix,k=16,size=325,node_order=[]):
    if not os.path.exists(prefix+"/La"):
        os.makedirs(prefix+"/La")
    for i in range(0,k):
            A=np.load(prefix+"adj/A"+str(i)+".npy")
            if A.max()==0.0:
                A=np.diag(np.ones(A.shape[0], dtype=np.float32))
            else:
                I = np.diag(np.ones(A.shape[0], dtype=np.float32))
                A = np.mat(A+I)
                A = A.A

            D_diag = A.sum(axis=1)
            D_ = np.diag(np.power(D_diag, -1))
            D_[np.isinf(D_)] = 0.

            LA1 = 0.5  * (np.matmul(D_, A))
            np.save(prefix+"/La/LA" + str(i), LA1)

    for i in range(0,k):
        for j in range(0,k):
          if i!=j:
            A=np.load(prefix+"/adj/A"+str(i)+"-A"+str(j)+".npy")
            if A.max() == 0.0:
                LA1 = A
                np.save(prefix+"/La/LA" + str(i)+"-LA" + str(j), LA1)
                continue
            A = np.mat(A)
            A=A.A
            I = np.diag(np.ones(A.shape[0], dtype=np.float32))
            D_diag = A.sum(axis=1)
            D_ = np.diag(np.power(D_diag, -1 ))
            D_[D_ == np.inf] = 0
            LA1=0.5/(k-1)*(np.matmul(D_, A))
            np.save(prefix+"/La/LA" + str(i)+"-LA" + str(j), LA1)


    P=np.zeros((size, size))
    LAlengthlist=[]
    for i in range(0,k):
        A1=np.load(prefix+"/La/LA"+str(i)+".npy")
        LAlengthlist.append(A1.shape[0])
    for i in range(0,k-1):
            LAlengthlist[i+1]+= LAlengthlist[i]
    for i in range(0,k):
            A1=np.load(prefix+"/La/LA"+str(i)+".npy")
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
                A1 = np.load(prefix+"/La/LA" + str(i) + "-LA" + str(j) + ".npy")
                shape = A1.shape
                for x in range(0, shape[0]):
                    for y in range(0, shape[1]):
                        if i == 0:
                            P[x, (LAlengthlist[j - 1])+ y] = A1[x, y]
                        elif j==0:
                            P[(LAlengthlist[i - 1])+x, y] = A1[x, y]
                        else:
                            P[(LAlengthlist[i - 1]) + x, (LAlengthlist[j - 1]) + y] = A1[x, y]

    P = np.mat(P,dtype=np.float32)
    I = np.diag(np.ones(P.shape[0], dtype=np.float32))
    LP=I-(cal_gcn_matrix(P)+cal_gcn_matrix_T(P))/2
    shutil.rmtree(prefix+"/adj")
    shutil.rmtree(prefix+"/La")

    LA_order_list=node_order
    OLA=np.zeros((LP.shape[0],LP.shape[1]),dtype=np.float32)
    for i in range(0,len(LA_order_list)):
            OLA[LA_order_list[i]]=LP[i]
    return np.mat(OLA)







