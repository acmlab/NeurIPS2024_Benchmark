import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import KFold
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as pyg_Dataloader
from models import get_model

def train_seq(model, dataset, optimizer, criterion, device):
        batchsize = 64
        loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=8)
        model.train()
        losses = []
        bar = tqdm(enumerate(loader), total=len(loader))
        for batch_idx, (input_embedding, target) in bar:
            logits,_ = model(input_embedding.to(device)) 
            logits = logits.unsqueeze(0) if logits.dim() == 1 else logits

            loss = criterion(logits, target.to(device)) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return sum(losses)/len(losses)
    
def test_seq(model, dataset, device):
    batchsize = 64
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=8)
    model.eval()
    preds = []
    gts = []
    bar = tqdm(enumerate(loader), total=len(loader))
    for batch_idx, (input_embedding, target) in bar:
        logits,feas = model(input_embedding.to(device))
        logits = logits.unsqueeze(0) if logits.dim() == 1 else logits
        
        pred = logits.argmax(dim=-1) 
        preds.append(pred.cpu().numpy())
        gts.append(target.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)

    accuracy = accuracy_score(gts, preds)
    pre = precision_score(gts, preds, average='weighted')
    rec = recall_score(gts, preds, average='weighted')
    f1 = f1_score(gts, preds, average='weighted')
    
    return accuracy, pre, rec, f1

def train_spa(model, dataset, criterion, optimizer, device):
    batchsize = 64
    loader = pyg_Dataloader(dataset, batch_size=batchsize, shuffle=True, num_workers=8)
    model.train()
    losses = []
    for data in loader:  
        data = data.to(device)
        out,_ = model(data)  
        loss = criterion(out, data.y)
        losses.append(loss.item())
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad()  
    return sum(losses)/len(losses)
        
        
def test_spa(model, dataset, device):
    batchsize = 64
    loader = pyg_Dataloader(dataset, batch_size=batchsize, shuffle=True, num_workers=8)
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for data in loader:  
            data = data.to(device)
            out,feas = model(data)  
            pred = out.argmax(dim=-1)  
            preds.append(pred.detach().cpu().numpy())
            gts.append(data.y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    accuracy = accuracy_score(gts, preds)
    pre = precision_score(gts, preds, average='weighted')
    rec = recall_score(gts, preds, average='weighted')
    f1 = f1_score(gts, preds, average='weighted')

    return accuracy, pre, rec, f1


def run_seq(model, train, val, test, device, total_epoch):
    model = model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
    best_val_acc = 0
    for epoch in range(1, 1+total_epoch):
        loss = train_seq(model, train, optimizer, criterion, device)
        val_acc, _, _, _ = test(val)
        test_acc, pre, rec, f1 = test(test)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        print(f'Epoch: {epoch:03d}, best Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f},pre: {pre:.4f}, rec: {rec:.4f}, f1: {f1:.4f}, loss: {loss:.4f}')
        
def run_spa(model, train, val, test, device, total_epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
    best_val_acc = 0
    for epoch in range(1, 1+total_epoch):
        loss = train_seq(model, train, optimizer, criterion, device)
        val_acc, _, _, _ = test(val)
        test_acc, pre, rec, f1 = test(test)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        print(f'Epoch: {epoch:03d}, best Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f},pre: {pre:.4f}, rec: {rec:.4f}, f1: {f1:.4f}, loss: {loss:.4f}')


def kFold_seq(model1, splits, dataset, device, total_epoch):
    acc = []
    precision = []
    f1_scores = []
    kf = KFold(n_splits=splits, shuffle=False, random_state=None)
    for i, (train_index, test_index) in enumerate(kf.split(dataset)):
        train_dataset = [dataset[idx] for idx in train_index]
        test_dataset = [dataset[idx] for idx in test_index]
<<<<<<< HEAD

=======
            
>>>>>>> 52cd94670142a524ae166ca548b5c66f07ac4500
        model = copy.deepcopy(model1).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)

        best_acc = 0
        best_pre = 0
        best_f1 = 0
        for epoch in range(1, 1+total_epoch):
            loss = train_seq(model, train_dataset, optimizer, criterion, device)
            test_acc, pre, rec, f1 = test_seq(model, test_dataset, device)
            if test_acc > best_acc:
                best_acc = test_acc
                best_pre = pre
                best_f1 = f1
            print(f'Epoch: {epoch:03d}, best Acc: {best_acc:.4f}, Test Acc: {test_acc:.4f}, pre: {pre:.4f}, rec: {rec:.4f}, f1: {f1:.4f}, loss: {loss:.4f}')
        acc.append(best_acc)
        precision.append(best_pre)
        f1_scores.append(best_f1)

        print(f"Fold {i}:", best_acc, best_pre, best_f1)

    total_acc = sum(acc)/splits
    total_pre = sum(precision)/splits
    total_f1 = sum(f1_scores)/splits

    print("Total_acc:", total_acc, "Total_pre:", total_pre, "Total_f1:", total_f1)   
    print("parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))     


def kFold_spa(model1, splits, dataset, device, total_epoch):
    acc = []
    precision = []
    f1_scores = []
    kf = KFold(n_splits=splits, shuffle=False, random_state=None)
    for i, (train_index, test_index) in enumerate(kf.split(dataset)):
        train_dataset = [dataset[idx] for idx in train_index]
        test_dataset = [dataset[idx] for idx in test_index]
<<<<<<< HEAD

=======
            
>>>>>>> 52cd94670142a524ae166ca548b5c66f07ac4500
        model = copy.deepcopy(model1).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
        
        best_test_acc = 0
        best_pre = 0
        best_f1 = 0
        for epoch in range(1, 1+total_epoch):
            loss = train_spa(model, train_dataset, criterion, optimizer, device)
            test_acc, pre, rec, f1 = test_spa(model, test_dataset, device)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_pre = pre
                best_f1 = f1
            # print(f'Epoch: {epoch:03d},  Test Acc: {test_acc:.4f}, Loss: {loss:.4f}, pre: {pre:.4f}, rec: {rec:.4f}, f1: {f1:.4f}')
            
        acc.append(best_test_acc)
        precision.append(best_pre)
        f1_scores.append(best_f1)
        print(f"Fold {i}:", best_test_acc, best_pre, best_f1)
    total_acc = sum(acc)/splits
    total_pre = sum(precision)/splits
    total_f1 = sum(f1_scores)/splits

    print("Total_acc:", total_acc, "Total_pre:", total_pre, "Total_f1:", total_f1)
    print("parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
