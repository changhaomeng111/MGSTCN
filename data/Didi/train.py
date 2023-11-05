import argparse
import tensorflow as tf
import os
import yaml
from scripts import Supra_adjacency,Supra_Laplace
import  pandas as pd
from model.supervisor import MGSTCNSupervisor
import numpy as np

def main(args):
        with open(args.config_filename) as f:
            supervisor_config = yaml.safe_load(f)

            DiverseGraphs=[]

            sz_flow = pd.read_csv(r'./data/didi/adjacentMatrix.csv', sep=',', header=None, engine='python')
            dat_1 = np.mat(sz_flow,dtype='float32')
            np.save('didiadj', dat_1)
            size=dat_1.shape[0]
            print(size)
            
            for l in [16]:
                node_order=Supra_adjacency.construct_supra_adjacency(l=l,adj_npy=dat_1)
                LA=Supra_Laplace.construct_supra_Laplace("./data/didi/", k = l, size = size,node_order=node_order)
                LA[LA<0]=0
                np.save('LA'+str(l),LA)
                DiverseGraphs.append(LA)
            #LA=np.load('LA16.npy')
            DiverseGraphs.append(LA)
            config = tf.ConfigProto(allow_soft_placement=True)
            os.environ["CUDA_VISIBLE_DEVICES"] = '1'
            config.gpu_options.allow_growth = False  # 程序按需申请内存

            with tf.Session(config=config) as sess:
                supervisor = MGSTCNSupervisor(adj_mx=DiverseGraphs, **supervisor_config)
                supervisor.train(sess=sess)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_filename', default='./data/didi/MGSTCN_train.yaml', type=str)
    parser.add_argument('--use_cpu_only', default=False, type=bool)
    args = parser.parse_args()
    main(args)
