import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from dataset import load
import argparse
from model import GrokFormer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import torchmetrics



def main_run(dataset, gpu, adj, E, U, feat, labels,epoch, batch,seed):
    device = 'cuda:{}'.format(gpu)
    torch.cuda.set_device(gpu)
    nb_epochs = epoch
    batch_size = batch
    patience = 20

    lr = args.lr
    l2_coef = args.weight_decay

    print(adj.shape)
    ft_size = feat[0].shape[1]
    max_nodes = feat[0].shape[0]
    num_classes = len(torch.unique(labels))
    #model = GCN(ft_size, num_classes,args)
    model = GrokFormer(num_classes,ft_size,args)
    #model = NAGformer(ft_size, num_classes,args)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    model.to(device)
    cnt_wait = 0
    best = 1e9
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    accuracies = []
    itr = (adj.shape[0] // batch_size) + 1
    
    for train_idx, test_idx in kf.split(np.zeros(len(labels.cpu().numpy())), labels.cpu().numpy()):
        for epoch in range(nb_epochs):
            epoch_loss = 0.0

            for idx in range(0, len(train_idx), batch_size):
                model.train()
                optimiser.zero_grad()
                batch = train_idx[idx: idx + batch_size]
                out = model(E[batch],U[batch],feat[batch])
                nll = F.nll_loss(out, labels[batch])
                epoch_loss += nll
                nll.backward()
                optimiser.step()

            epoch_loss /= itr

            if epoch_loss < best:
                best = epoch_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), f'{dataset}-{gpu}.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == patience:
                break

        model.load_state_dict(torch.load(f'{dataset}-{gpu}.pkl'))     
        embeds = model(E[test_idx],U[test_idx],feat[test_idx])
        
        pred_labels = torch.argmax(embeds, dim=1)
    
        # 2. 移到 CPU 并转 NumPy
        pred_labels_np = pred_labels.cpu().numpy()
        true_labels_np = labels[test_idx].cpu().numpy()
    
        # 3. 计算 ACC
        
        acc = np.mean(pred_labels_np == true_labels_np)
        print('test_acc:{}'.format(acc))
        accuracies.append(acc)

    print('10 runs mean acc:{}'.format(np.mean(accuracies)))


    f = open("./res/{}.txt".format(dataset), 'a')
    f.write(
        "epoch:{},batch_size:{},lr:{}, wd:{} ,hid:{},dim:{},k:{},nlayer:{}, tran_dropout:{},feat_dropout:{},prop_dropout:{},nhead:{}, acc_test:{},std:{}".format(args.epoch,args.batch_size,args.lr, args.weight_decay,
                                                                                                  args.hidden_dim,
                                                                                                  args.dim,
                                                                                                  args.k,
                                                                                                  args.nlayer,
                                                                                                  args.tran_dropout,
                                                                                                  args.feat_dropout,
                                                                                                  args.prop_dropout,
                                                                                                  args.nheads,
                                                                                                  np.mean(accuracies),
                                                                                                  np.std(
                                                                                                      accuracies) * 100,
                                                                                                  ))
    f.write("\n")
    f.close()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--device', type=int, default=2, help='GPU device.')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--nlayer', type=int, default=1)
    parser.add_argument('--tran_dropout', type=float, default=0.0)
    parser.add_argument('--feat_dropout', type=float, default=0.0)
    parser.add_argument('--prop_dropout', type=float, default=0.0)
    parser.add_argument('--nheads', type=float, default=1)
    parser.add_argument('--dataset', default='IMDB-BINARY')
    parser.add_argument('--k', type=int, default=3,help='Number of Transformer layers')

    args = parser.parse_args()
    print(args)
    # gpu = 1
    device = 'cuda:{}'.format(args.device)
    torch.cuda.set_device(args.device)

    adj, diff, feat, labels, num_nodes, E, U = load(args.dataset)

    feat = torch.FloatTensor(feat).to(device)
    diff = torch.FloatTensor(diff).to(device)
    adj = torch.FloatTensor(adj).to(device)
    labels = torch.LongTensor(labels).to(device)
    E = torch.FloatTensor(E).to(device)
    U = torch.FloatTensor(U).to(device)

    epoch = 200
    batch = 128
    print(f'####################{args.dataset}####################')
    seed = args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    main_run(args.dataset,args.device,adj, E,U,feat,labels, epoch, batch, seed)
    print('################################################')
