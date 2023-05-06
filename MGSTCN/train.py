import argparse
import tensorflow as tf
import os
import yaml
from scripts import Supra_adjacency,Supra_Laplace
import numpy as np
from model.supervisor import MGSTCNSupervisor
import time

def main(args):
        with open(args.config_filename) as f:
            supervisor_config = yaml.safe_load(f)

            DiverseGraphs=[]

            num_sensors = 170
            dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
            dist_mx[:] = np.inf
            with open("data/PEMS08/PEMS08.csv") as text:
                i = 0
                for line in text:
                    if i == 0:
                        i += 1
                        continue
                    vertices = line.strip().split(",")
                    source = int(vertices[0])
                    target = int(vertices[1])
                    link_weight = float(vertices[2])
                    dist_mx[source, target] = link_weight


            for l in [8]:  #1,8,16,32,64
                node_order=Supra_adjacency.construct_supra_adjacency("./data/PEMS08/",l=l,adj_npy=dist_mx)
                LA=Supra_Laplace.construct_supra_Laplace("./data/PEMS08/", k = l, size = num_sensors,node_order=node_order)
                np.save('La'+str(l),LA)
                DiverseGraphs.append(LA)

            config = tf.ConfigProto(allow_soft_placement=True)
            os.environ["CUDA_VISIBLE_DEVICES"] = '1'
            config.gpu_options.allow_growth = False  # 程序按需申请内存

            with tf.Session(config=config) as sess:
                supervisor = MGSTCNSupervisor(adj_mx=DiverseGraphs, **supervisor_config)
                supervisor.train(sess=sess)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_filename', default='./data/PEMS/MGSTCN_train.yaml', type=str)
    parser.add_argument('--use_cpu_only', default=False, type=bool)
    args = parser.parse_args()
    main(args)
