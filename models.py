import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool, GINConv, global_add_pool
from torch_geometric.nn import MLP as pyg_MLP
import argparse

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.lin1 = Linear(hidden_size, in_feats)
        self.lin2 = Linear(in_feats, out_feats)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x.float(), edge_index, edge_weight)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch) 
        x = F.dropout(x, p=0.3, training=self.training)
        x_fea = self.lin1(x)
        x = F.relu(x_fea)
        x = self.lin2(x)
        return x, x_fea
    
class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = pyg_MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = pyg_MLP([hidden_channels, hidden_channels, out_channels], norm=None, dropout=0.5)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_add_pool(x, data.batch)
        return self.mlp(x), x
    


class Positional_Encoding(nn.Module):
    def __init__(self, d_model, full_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / d_model)) for i in range(d_model)] for pos in range(full_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out

class Embedding(nn.Module):
    def __init__(self, in_feas, d_model, len, device):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Linear(in_feas, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.postion_embedding = Positional_Encoding(d_model, len+1, 0.5, device)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, self.tok_embed(x)), dim=1)
        return self.postion_embedding(x)
    
class TF(nn.Module):
    def __init__(self, in_feas, d_model, n_heads, d_ff, len, num_layers, num_classes, device):
        super(TF, self).__init__()
        self.embedding = Embedding(in_feas, d_model, len, device)
        encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_ff,
        dropout=0.2,
        activation='relu',
        batch_first=True, 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc1 = nn.Linear(d_model, in_feas)
        self.cls_head = nn.Sequential(
                                    #   nn.LayerNorm(d_model),
                                      nn.Linear(in_feas, num_classes)
                                      )

    def forward(self, input_ids):
        output = self.embedding(input_ids)
        output = self.transformer_encoder(output)
        x = self.fc1(output[:, 0])
        out=F.relu(x)
        pred_clf = self.cls_head(out)
        return pred_clf, x

class Mixer_Embedding(nn.Module):
    def __init__(self, input_dim):
        super(Mixer_Embedding, self).__init__()
        self.tok_embed = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.tok_embed(x)

class MixerBlock(nn.Module):
    def __init__(self, input_dim, length, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(input_dim),
            Rearrange('b n d -> b d n'),
            nn.Linear(length, tokens_mlp_dim),
            nn.ReLU(),
            nn.Linear(tokens_mlp_dim, length),
            Rearrange('b d n -> b n d')
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, channels_mlp_dim),
            nn.ReLU(),
            nn.Linear(channels_mlp_dim, input_dim)
        )

    def forward(self, x):
        x = x + self.token_mixing(x)
        x = x + self.channel_mixing(x)
        return x

class MLP_Mixer(nn.Module):
    def __init__(self, input_dim, length, tokens_mlp_dim, channels_mlp_dim, num_classes, num_blocks):
        super(MLP_Mixer, self).__init__()
        self.embedding = Mixer_Embedding(input_dim)
        self.input_dim = input_dim
        self.num_blocks = num_blocks
        self.mixer_blocks = nn.ModuleList([MixerBlock(input_dim, length, tokens_mlp_dim, channels_mlp_dim) for _ in range(num_blocks)])
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, data):
        x = self.embedding(data)
        for block in self.mixer_blocks:
            x = block(x)
        x = x.mean(dim=1)  
        out = self.fc(x)
        return out, x


class Temporal_Conv(nn.Module):
    def __init__(self, in_channels, filters, k, d, activation):
        super(Temporal_Conv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, filters, k, dilation=d, padding=0)
        self.activation = activation
        self.bn = nn.BatchNorm1d(filters)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn(h)
        h = self.activation(h)
        h = self.pool(h)
        return h
    
class CNN_1D(nn.Module):
    def __init__(self, nrois, f1, f2, dilation_exponential, k1, dropout, readout, num_classes):
        super(CNN_1D, self).__init__()
        self.readout = readout
        self.layer0 = Temporal_Conv(nrois, f1, k1, dilation_exponential, F.relu)
        self.layer1 = Temporal_Conv(f1, f2, k1, dilation_exponential, F.relu)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.max = nn.AdaptiveMaxPool1d(1)
        dim = 2 if readout == 'meanmax' else 1
        self.drop = nn.Dropout(p=dropout)
        self.classify = nn.Linear(f2*dim, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h0 = self.layer0(x) 
        h1 = self.layer1(h0)
        h_avg = torch.squeeze(self.avg(h1))
        h_max = torch.squeeze(self.max(h1))
        if self.readout == 'meanmax':
            h = torch.cat((h_avg, h_max),1)
        else:
            h = h_avg
        h = self.drop(h)
        hg = self.classify(h)
        return hg, h

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, input_size)
        self.fc = nn.Linear(input_size, num_classes)
        self.device = device
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        x = self.fc1(out[:, -1, :])
        pre = F.relu(x)
        pre = self.fc(pre)
        # pre = self.fc(out.mean(dim=1))
        return pre, x
    
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, input_size)
        self.fc = nn.Linear(input_size, num_classes)
        self.device = device
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.rnn(x, h0)
        x = self.fc1(out[:, -1, :])
        pre = F.relu(x)
        pre = self.fc(pre)
        # pre = self.fc(out.mean(dim=1))
        return pre, x
    
def lower_triangular(tensor):
    n = tensor.size(0)
    indices = torch.tril_indices(n, n, offset=0)
    return tensor[indices[0], indices[1]] 
  
class MLP(nn.Module):
    def __init__(self, nrois, f1, f2, num_classes, device):
        super(MLP, self).__init__()
        self.in_dim = int(nrois * (nrois + 1) / 2)
        self.layer0 = Sequential(Linear(self.in_dim, f1), BatchNorm1d(f1), ReLU())
        self.fc1 = Linear(f1, f2)
        self.layer1 = Sequential(BatchNorm1d(f2), ReLU())
        self.drop = nn.Dropout(p=0.5)
        self.classify = nn.Linear(f2, num_classes)
        self.device = device

    def forward(self, data):
        b, t, r = data.shape
        data_mlp = torch.zeros(b, self.in_dim).to(self.device)
        for bs in range(b):
            # tmp = torch.corrcoef(data[bs].squeeze().T)
            data_mlp[bs] = lower_triangular(data[bs])
        h0 = self.layer0(data_mlp)
        x = self.fc1(h0)
        h1 = self.layer1(x)
        h = self.drop(h1)
        hg = self.classify(h)

        return hg, x
    

def get_model_params(dataset_name):
    hidden_dim=1024
    num_classes=2
    if dataset_name in ['ADNI', 'PPMI', 'ABIDE']:
        input_dim = 116
        times = 300
        if dataset_name == 'ADNI':
            hidden_dim = 512
            times = 140
        if dataset_name == 'PPMI':
            num_classes = 4
            times = 240
    elif dataset_name == 'OASIS':
        input_dim = 160
        times = 2000
    else:
        input_dim = 360
        if 'WM' in dataset_name:
            num_classes = 8
            times = 39
        else:
            num_classes = 7
            times = 176
    return input_dim, hidden_dim, num_classes, times


def get_model(model_name, dataset_name, device):
    input_dim, hidden_dim, num_classes, times = get_model_params(dataset_name)
    models_dict = {
        'GCN': GCN(input_dim, hidden_dim, num_classes),
        'GIN': GIN(input_dim, hidden_dim, num_classes, num_layers=2),
        'MGNN': GIN(input_dim+45, hidden_dim, num_classes, num_layers=2),
        'MLP': MLP(nrois=input_dim, f1=hidden_dim, f2=input_dim, num_classes=num_classes, device=device),
        'CNN': CNN_1D(nrois=input_dim, f1=hidden_dim, f2=input_dim, dilation_exponential=2, k1=3, dropout=0.2, readout='mean', num_classes=num_classes),
        'RNN': RNNClassifier(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, num_classes=num_classes, device=device),
        'LSTM': LSTMClassifier(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, num_classes=num_classes, device=device),
        'MIXER': MLP_Mixer(input_dim=input_dim, length=times, tokens_mlp_dim=256, channels_mlp_dim=hidden_dim, num_classes=num_classes, num_blocks=4),
        'TF': TF(in_feas=input_dim, d_model=hidden_dim, n_heads=2, d_ff=hidden_dim * 4, len=times, num_layers=4, num_classes=num_classes, device=device)
    }
    if model_name in models_dict:
        return models_dict[model_name]
    else:
        raise argparse.ArgumentTypeError(f"Unsupported model: {model_name}")