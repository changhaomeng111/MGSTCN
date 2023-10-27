import argparse
import tensorflow as tf
import os
import yaml

from model.supervisor import MGSTCNSupervisor
import numpy as np

def main(args):
        with open(args.config_filename) as f:
            supervisor_config = yaml.safe_load(f)
            LA =  np.mat(np.load("./data/PEMS-BAY/OLA64.npy"))
            LA[LA<0]=0
            config = tf.ConfigProto(allow_soft_placement=True)
            os.environ["CUDA_VISIBLE_DEVICES"] = '3'
            config.gpu_options.allow_growth = False  # 程序按需申请内存

            with tf.Session(config=config) as sess:
                supervisor = MGSTCNSupervisor(adj_mx=LA, **supervisor_config)
                supervisor.train(sess=sess)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_filename', default='data/PEMS-BAY/MGSTCN_train.yaml', type=str)
    parser.add_argument('--use_cpu_only', default=False, type=bool)
    args = parser.parse_args()
    main(args)
