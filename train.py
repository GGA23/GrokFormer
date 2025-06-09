import time
import yaml
import copy
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score, r2_score
from model import GrokFormer
from utils import count_parameters, init_params, seed_everything, get_split
from tqdm import tqdm

def main_worker(args):
    #print(args)
    seed_everything(args.seed)
    device = 'cuda:{}'.format(args.cuda)
    torch.cuda.set_device(args.cuda)

    
    e, u, x, y = torch.load('../data/{}.pt'.format(args.dataset))
    
    if len(y.size()) > 1:
        if y.size(1) > 1:
            nclass =  y.size(1)    
            y = torch.argmax(y, dim=1)
        else:
            y = y.view(-1)
            nclass = y.max().item() + 1
            
    e, u, x, y = e.to(device), u.to(device), x.to(device), y.to(device)

    train, valid, test = get_split(args.dataset, y, nclass, args.seed) 
    train, valid, test = map(torch.LongTensor, (train, valid, test))
    train, valid, test = train.to(device), valid.to(device), test.to(device)

    nfeat = x.size(1)
     
    net = GrokFormer(nclass, nfeat, args.nlayer, args.hidden_dim, args.dim, args.nheads, args.k, args.tran_dropout, args.feat_dropout, args.prop_dropout,args.norm).to(device)
    net.apply(init_params)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(count_parameters(net))

    res = []
    min_loss = 100.0

    max_acc1 = 0
    max_acc2 = 0
    counter = 0
    best_val_acc = 0
    best_test_acc = 0
    
    time_run=[]
    for idx in range(args.epoch):
        t_st=time.time()
        net.train()
        optimizer.zero_grad()
        logits = net(e, u, x)
        loss = F.cross_entropy(logits[train], y[train])
        loss.backward()
        optimizer.step()

        time_epoch = time.time()-t_st  # each epoch train times
        time_run.append(time_epoch)

        net.eval()
        logits = net(e, u, x)

        evaluation = torchmetrics.Accuracy(task='multiclass', num_classes=nclass)
        val_loss = F.cross_entropy(logits[valid], y[valid]).item()
        val_acc = evaluation(logits[valid].cpu(), y[valid].cpu()).item()
        test_acc = evaluation(logits[test].cpu(), y[test].cpu()).item()
        res.append([val_loss, val_acc, test_acc])
        
        if val_loss < min_loss:
            min_loss = val_loss
            best_val_acc = val_acc
            best_test_acc = test_acc
            counter = 0
                
        else:
            counter += 1

        if counter == 200:
            max_acc1 = sorted(res, key=lambda x: x[0], reverse=False)[0][-1]
            max_acc2 = sorted(res, key=lambda x: x[1], reverse=True)[0][-1]
            print(max_acc1, max_acc2)
            break

        
    return max_acc1,max_acc2, time_run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=3)
    parser.add_argument('--runs', type=int, default=10) #5 for penn
    parser.add_argument('--dataset', default='physics')
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--nheads', type=int, default=1)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--nlayer', type=int, default=1)
    parser.add_argument('--tran_dropout', type=float, default=0.3, help='dropout for neural networks.')
    parser.add_argument('--feat_dropout', type=float, default=0.3, help='dropout for neural networks.')
    parser.add_argument('--prop_dropout', type=float, default=0.7, help='dropout for neural networks.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='weight decay.')
    parser.add_argument('--norm', default='none')

    args = parser.parse_args()


    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
 
    results=[]
    time_results=[]
    SEEDS = np.arange(1, 11)
    for RP in tqdm(range(args.runs)):
        args.seed = SEEDS[RP]
        set_seed(args.seed)
        best_val_acc, best_test_acc, time_run = main_worker(args)
        results.append(best_test_acc)
        time_results.append(time_run)

    run_sum=0
    epochsss=0
    for i in time_results:
        run_sum+=sum(i)
        epochsss+=len(i)
    print("each run avg_time:",run_sum/(args.runs),"s")
    print("each epoch avg_time:",1000*run_sum/epochsss,"ms")  
    print(np.mean(results))
