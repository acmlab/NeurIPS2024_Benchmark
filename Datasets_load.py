import os
import numpy as np
import pandas as pd
import re
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
from help_funs import *


class Dataset_WM(Dataset):
    def __init__(self, data_dir, label_dir, spatial=False, graph=False, mgnn=False):
        super(Dataset_WM, self).__init__()
        self.spatial = spatial
        self.graph = graph
        self.mgnn = mgnn
        self.base_path = data_dir
        self.data_path = [os.path.join(self.base_path, name) for name in
                          sorted_aphanumeric(os.listdir(data_dir)) if name.endswith('.txt') or name.endswith('.csv')]
        self.data = [np.loadtxt(path, delimiter='\t', dtype=np.float32) for path in self.data_path]

        self.label_path = label_dir
        self.labels_path = [os.path.join(self.label_path, name) for name in
                          sorted_aphanumeric(os.listdir(label_dir)) if name.endswith('.txt') or name.endswith('.csv')]
        self.labels = [np.loadtxt(path, delimiter=',', dtype=np.float32) for path in self.labels_path]
        self.labels = self.labels * len(self.data)

        self.temp_label = []
        self.temp_clip = []

        for sample, label in zip(self.data, self.labels):
            label_pos = self.pick_labels(label)
            batch_data = sample.squeeze()
            for ll in label_pos:
                self.temp_label.append(int(ll-1))
                self.temp_clip.append(batch_data[label == ll])

        self.data = self.temp_clip
        self.labels = self.temp_label

        self.n_data = len(self.data)
        print(self.n_data)
        print(data_dir)
        print(f'{self.data_path[0]},...,{self.data_path[-1]}')

    def __len__(self):
        return self.n_data

    def __add__(self, other):
        self.data_path += other.data_path
        self.labels = np.concatenate((self.labels, other.labels), axis=0)
        self.n_data += other.n_data
        return self

    def __getitem__(self, idx):

        data = self.data[idx]
        label = self.labels[idx]
        if self.spatial:
            data = torch.corrcoef(torch.tensor(data).squeeze().T)
            data = torch.nan_to_num(data)
            if self.graph:
                data = pearson_dataset(data, label, 10, self.mgnn)
                return data
        return data, label

    def pick_labels(self, tensor):
        arr = np.unique(tensor)
        mask = np.ones(len(arr), dtype=bool)
        mask[0] = False  # resting
        result = arr[mask]
        assert len(result) == 8
        return result

class Dataset_tasks(Dataset):
    def __init__(self, data_dir, spatial=False, graph=False, mgnn=False):
        super(Dataset_tasks, self).__init__()
        self.graph = graph
        self.spatial = spatial
        self.mgnn = mgnn
        self.base_path = data_dir
        self.data_path = [os.path.join(self.base_path, name) for name in
                          sorted_aphanumeric(os.listdir(data_dir)) if name.endswith('.txt') or name.endswith('.csv')] 

        self.labels = []
        for name in sorted_aphanumeric(os.listdir(data_dir)):
            label_class = name.split("_")[1]
            if label_class =='EMOTION':
                self.labels.append(0)
            if label_class =='GAMBLING':
                self.labels.append(1)
            if label_class =='LANGUAGE':
                self.labels.append(2)
            if label_class =='MOTOR':
                self.labels.append(3)
            if label_class =='RELATIONAL':
                self.labels.append(4)
            if label_class =='SOCIAL':
                self.labels.append(5)
            if label_class =='WM':
                self.labels.append(6)         

        self.n_data = len(self.data_path)
        print(self.n_data)
        print(data_dir)
        print(f'{self.data_path[0]},...,{self.data_path[-1]}')

    def __len__(self):
        return self.n_data

    def __add__(self, other):
        self.data_path += other.data_path
        self.labels = np.concatenate((self.labels, other.labels), axis=0)
        self.n_data += other.n_data
        return self

    def __getitem__(self, idx):
        path = self.data_path[idx]
        data = np.loadtxt(path, delimiter='\t', dtype=np.float32)[:176]
        label = self.labels[idx]
        if self.spatial:
            data = torch.corrcoef(torch.tensor(data).squeeze().T)
            data = torch.nan_to_num(data)
            if self.graph:
                data = pearson_dataset(data, label, 10, self.mgnn)
                return data
        return data, label

    
class Dataset_ADNI(Dataset):
    def __init__(self, data_dir, label_file, spatial=False, graph=False, mgnn=False):
        super(Dataset_ADNI, self).__init__()
        self.graph = graph
        self.spatial = spatial
        self.mgnn = mgnn
        self.data_dir = data_dir
        self.label_file = label_file
        self.data = []
        self.labels = []
        self.load_data()
        # self.pad_sentences()

    def load_data(self):
        sentence_sizes = []
        labels_df = pd.read_csv(self.label_file, header=0)

        for filename in os.listdir(self.data_dir):
            if filename.startswith('sub_'):
                id = filename.split('_')[1]

                label_row = labels_df[labels_df['subject_id'] == id]
                if not label_row.empty:
                    label = label_row.iloc[0]['DX']
                    if label in ['CN', 'SMC', 'EMCI']:
                        self.labels.append(0)
                    elif label in ['LMCI', 'AD']:
                        self.labels.append(1)
                    else:
                        print('Label Error')
                        self.labels.append(-1)
                    features = np.loadtxt(os.path.join(self.data_dir, filename))
                    self.data.append(features)
                    sentence_sizes.append(features.shape[0])
        if self.spatial == False:
            self.max_sentence_size = max(sentence_sizes)

    def pad_sentences(self):
        self.data = [torch.cat((torch.tensor(sentence), torch.zeros(self.max_sentence_size - sentence.shape[0], sentence.shape[1])), dim=0) for sentence in self.data]        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx]).float()
        label = self.labels[idx]
        if self.spatial:
            data = torch.corrcoef(data.squeeze().T)  
            data = torch.nan_to_num(data)
            if self.graph:
                data = pearson_dataset(data, label, 10, self.mgnn)
                return data
        else:
            data = data[:140].float()

        return data, label

class Dataset_PPMI(Dataset):
    def __init__(self, root_dir, spatial=False, graph=False, mgnn=False):
        super(Dataset_PPMI, self).__init__()
        self.graph = graph
        self.spatial = spatial
        self.mgnn = mgnn
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.load_data()
        if self.spatial == False:
            self.pad_sentences()

    def load_data(self):
        sentence_sizes = []
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if 'AAL116_features_timeseries' in file:
                    file_path = os.path.join(subdir, file)
                    data = loadmat(file_path)
                    features = data['data']
                    sentence_sizes.append(features.shape[0]) 
                    label = self.get_label(subdir)
                    self.data.append(features)
                    self.labels.append(label)
        if self.spatial == False:            
            self.max_sentence_size = max(sentence_sizes)

    def get_label(self, subdir):
        if 'control' in subdir:
            return 0
        elif 'patient' in subdir:
            return 1
        elif 'prodromal' in subdir:
            return 2
        elif 'swedd' in subdir:
            return 3
        else:
            print("Label error")
            return -1 
        
    def pad_sentences(self):
        self.data = [torch.cat((torch.tensor(sentence), torch.zeros(self.max_sentence_size - sentence.shape[0], sentence.shape[1])), dim=0) for sentence in self.data]        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx]).float()
        label = self.labels[idx]
        if self.spatial:
            data = torch.corrcoef(data.squeeze().T)
            data = torch.nan_to_num(data)
            if self.graph:
                data = pearson_dataset(data, label, 10, self.mgnn)
                return data
        return data, label


class Dataset_OASIS(Dataset):
    def __init__(self, root_dir, spatial=False, graph=False, mgnn=False):
        super(Dataset_OASIS, self).__init__()
        self.graph = graph
        self.spatial = spatial
        self.mgnn = mgnn
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.load_data()
        if self.spatial == False:
            self.pad_sentences()

    def load_data(self):
        sentence_sizes = []
        for filename in os.listdir(self.root_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.root_dir, filename)
                data = self.load_txt(filepath)
                if data.size(0) >= 100:
                    label = self.get_label(filename)
                    self.data.append(data)
                    sentence_sizes.append(data.shape[0]) 
                    self.labels.append(label)
        if self.spatial == False:
            self.max_sentence_size = max(sentence_sizes)

    def load_txt(self, filepath):
        with open(filepath, 'r') as file:
            data = [[float(num) for num in line.split()] for line in file.readlines()]
        return torch.tensor(data)

    def get_label(self, filename):
        if 'CN' in filename:
            return 0
        elif 'AD' in filename:
            return 1
        else:
            print("Label error")
            return -1
    
    def pad_sentences(self):
        self.data = [torch.cat((torch.tensor(sentence), torch.zeros(self.max_sentence_size - sentence.shape[0], sentence.shape[1])), dim=0) for sentence in self.data]      

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        if self.spatial:
            data = torch.corrcoef(torch.tensor(data).squeeze().T)
            data = torch.nan_to_num(data)
            if self.graph:
                data = pearson_dataset(data, label, 10, self.mgnn)
                return data
        return data, label


def get_dataset(model_name, dataset_name):
    sequential_models = ['CNN', 'RNN', 'LSTM', 'MIXER', 'TF']
    graph_models = ['GCN', 'GIN', 'MGNN']
    other_spatial = ['SPD', 'MLP']

    if 'TASK' in dataset_name:
        data_path = './data/HCP/'+dataset_name
        if model_name in sequential_models:
            dataset = Dataset_tasks(data_path)
        elif model_name in other_spatial:
            dataset = Dataset_tasks(data_path, spatial=True)
        else:
            if model_name == 'MGNN':
                dataset = Dataset_tasks(data_path, spatial=True, graph=True, mgnn=True) 
                return dataset
            dataset = Dataset_tasks(data_path, spatial=True, graph=True)

    elif  'WM' in dataset_name:
        data_path = './data/HCP/WM/'+dataset_name
        label_path = './data/HCP/WM/label/'+dataset_name
        if model_name in sequential_models:
            dataset = Dataset_WM(data_path, label_path)
        elif model_name in other_spatial:
            dataset = Dataset_WM(data_path, label_path, spatial=True)
        else:
            if model_name == 'MGNN':
                dataset = Dataset_WM(data_path, spatial=True, graph=True, mgnn=True) 
                return dataset
            dataset = Dataset_WM(data_path, label_path, spatial=True, graph=True)     

    elif dataset_name == 'ADNI':
        data_path = f'./data/ADNI/AAL90'
        label_path = './data/ADNI/label-2cls_new.csv'
        if model_name in sequential_models:
            dataset = Dataset_ADNI(data_path, label_path)
        elif model_name in other_spatial:
            dataset = Dataset_ADNI(data_path, label_path, spatial=True)
        else:
            if model_name == 'MGNN':
                dataset = Dataset_ADNI(data_path, label_path, spatial=True, graph=True, mgnn=True) 
                return dataset
            dataset = Dataset_ADNI(data_path, label_path, spatial=True, graph=True)    

    elif dataset_name in ['PPMI', 'ABIDE']:
        data_path = './data/'+dataset_name
        if model_name in sequential_models:
            dataset = Dataset_PPMI(data_path)
        elif model_name in other_spatial:
            dataset = Dataset_PPMI(data_path, spatial=True)
        else:
            if model_name == 'MGNN':
                dataset = Dataset_PPMI(data_path, spatial=True, graph=True, mgnn=True) 
                return dataset
            dataset = Dataset_PPMI(data_path, spatial=True, graph=True) 

    elif dataset_name == 'OASIS':
        data_path = './data/'+dataset_name
        if model_name in sequential_models:
            dataset = Dataset_OASIS(data_path)
        elif model_name in other_spatial:
            dataset = Dataset_OASIS(data_path, spatial=True)
        else:
            if model_name == 'MGNN':
                dataset = Dataset_OASIS(data_path, spatial=True, graph=True, mgnn=True) 
                return dataset
            dataset = Dataset_OASIS(data_path, spatial=True, graph=True) 

    return dataset
