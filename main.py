import random
import torch
import numpy as np
from torch.utils.data import random_split

from Datasets_load import *
from models import *
from help_train import *

import warnings
warnings.filterwarnings("ignore")
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='3')
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GIN', 'MGNN', 'MLP', 'CNN', 'RNN', 'LSTM', 'MIXER', 'TF'])
    parser.add_argument('--dataset', type=str, default='TASK')
    parser.add_argument('--epoch', type=int, default=300)
    return parser.parse_args()


def set_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(gpu_id):
    return torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')


def main():
    args = parse_args()
    set_seeds()
    epoch = args.epoch
    device = get_device(args.gpu)
    model = get_model(args.model, args.dataset, device).to(device)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}, parameters: {num_parameters}")

    seq_models = ['CNN', 'RNN', 'LSTM', 'MIXER', 'TF', 'MLP'] # MLP is the spatial model indeed, but uses the same train func as sequential models
    spa_models = ['GCN', 'GIN', 'MGNN']

    task_datasets = ['TASK', 'WM']
    disease_datasets = ['ADNI', 'OASIS', 'PPMI', 'ABIDE']

    if args.dataset in disease_datasets:
        whole_dataset = get_dataset(args.model, args.dataset)
        splits = 10 if args.dataset in ['ADNI', 'PPMI'] else 5
        if args.model in spa_models:
            kFold_spa(model, splits, whole_dataset, device, epoch)
        else:
            kFold_seq(model, splits, whole_dataset, device, epoch)
    
    else: 
        dataset_type = args.dataset
        train_dataset = get_dataset(args.model, dataset_type+'_LR') # Separated scan setting
        test_dataset = get_dataset(args.model, dataset_type+'_RL')
        len_val = len(test_dataset) * 0.4
        len_test = len(test_dataset) - len_val
        val_set, test_set = random_split(test_dataset, [len_val, len_test])

        if args.model in spa_models:
            run_spa(model, train_dataset, val_set, test_set, device, epoch)
        else:
            run_seq(model, train_dataset, val_set, test_set, device, epoch)

if __name__ == "__main__":
    main()

