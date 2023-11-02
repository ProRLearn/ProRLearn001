import pandas as pd
import numpy as np

import argparse
import torch

#=======================================================




if __name__ == '__main__':
    torch.munual_seed(10)
    np.random.seed(10)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True, help="dataset name")
    parser.add_argument('--input_dir', type=str, help='Input Directory of the parser',
                        default='dataset/Devign_utf8.csv')

    parser.add_argument('--Learning', type=float, help='Batch Size for training', default=2e-5)
    parser.add_argument('--seed',type=int, default = None, help='random seed')
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=16)
    args = parser.parse_args()

