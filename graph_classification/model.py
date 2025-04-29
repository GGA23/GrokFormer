import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_
from torch.nn.parameter import Parameter


class SineEncoding(nn.Module):
    def __init__(self, k=2, hidden_dim=64):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_ws = nn.ModuleList([nn.Linear(hidden_dim + 1, 1) for i in range(k)])
        self.k = k
        self.alpha = nn.Linear(self.k, 1, bias=False)

    def forward(self, e):
        # input:  [N]
        out_e = []
        ee = e.unsqueeze(-1)
        for i in range(self.k):
            eeig = torch.full(ee.shape, torch.tensor(1.0)).to(e.device)
            ei = ee.pow(i + 1)
            div = torch.FloatTensor(np.arange(1, int(self.hidden_dim/2) + 1)).to(e.device)
            pe = ei * div
            eeig = torch.cat((eeig, torch.sin(pe), torch.cos(pe)), dim=2)
            out_e.append(self.eig_ws[i](eeig))
        new_e = torch.cat(out_e, dim=2)
        new_e = self.alpha(new_e)
        return new_e
class FeedForwardNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_dropout_rate):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, int(num_heads * hidden_size))
        self.linear_k = nn.Linear(hidden_size, int(num_heads * hidden_size))
        self.linear_v = nn.Linear(hidden_size, int(num_heads * hidden_size))

        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(int(num_heads * hidden_size), hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        q = self.linear_q(q)  
        k = self.linear_k(k)
        v = self.linear_v(v)

        k = k.permute(0, 2, 1)  

        q = q * self.scale
        x = torch.bmm(q, k)  
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=1)
        x = self.att_dropout(x)
        x = torch.bmm(x,v)
        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class GrokFormer(nn.Module):

    def __init__(self, nclass, nfeat, args, norm='norm'):
        super(GrokFormer, self).__init__()

        self.norm = norm
        self.nfeat = nfeat
        self.nlayer = args.nlayer
        self.nheads = args.nheads
        self.hidden_dim = args.hidden_dim
        self.dim = args.dim

        self.feat_encoder = nn.Sequential(
            nn.Linear(nfeat, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, nclass),
        )


        self.linear_encoder = nn.Linear(nfeat, self.hidden_dim)

        self.eig_encoder = SineEncoding(args.k,self.dim)

        self.mha_dropout = nn.Dropout(args.tran_dropout)
        self.ffn_dropout = nn.Dropout(args.tran_dropout)
        self.prop_dropout = nn.Dropout(args.prop_dropout)
        
        self.k = args.k

        self.feat_dp1 = nn.Dropout(args.feat_dropout)
        self.feat_dp2 = nn.Dropout(args.feat_dropout)

        if norm == 'none':
            self.mha_norm = nn.LayerNorm(nclass)
            self.ffn_norm = nn.LayerNorm(nclass)
            self.mha = MultiHeadAttention(nclass, self.nheads, args.tran_dropout)
            self.ffn = FeedForwardNetwork(nclass, nclass, nclass)
        else:
            self.mha_norm = nn.LayerNorm(self.hidden_dim)
            self.ffn_norm = nn.LayerNorm(self.hidden_dim)
            self.mha = MultiHeadAttention(self.hidden_dim, self.nheads, args.tran_dropout)
            self.ffn = FeedForwardNetwork(self.hidden_dim, self.hidden_dim, self.hidden_dim)
            self.classify = nn.Linear(self.hidden_dim, nclass)


    def transformer_encoder(self, h, h_fur):
        mha_h = self.mha_norm(h)
        mha_h = self.mha(mha_h, mha_h, mha_h)
        mha_h_ = h + self.mha_dropout(mha_h) + h_fur

        ffn_h = self.ffn_norm(mha_h_)
        ffn_h = self.ffn(ffn_h)
        encoder_h = mha_h_ + self.ffn_dropout(ffn_h)
        return encoder_h

    def forward(self, e, u, x):
        N = e.size(0)
        ut = u.permute(0, 2, 1)

        if self.norm == 'none':
            h = self.feat_dp1(x)
            h = self.feat_encoder(h)
            h = self.feat_dp2(h)
        else:
            h = self.feat_dp1(x)
            h = self.linear_encoder(h)


        eig = self.eig_encoder(e)  # [N, d]
        new_e = eig

        for conv in range(self.nlayer):
            utx = ut @ h
            h_encoder = u @ (new_e * utx)
            h_encoder = self.prop_dropout(h_encoder)
            h = self.transformer_encoder(h,h_encoder)

        if self.norm == 'none':
            h = torch.max(h, dim=1)[0].squeeze()  # max pooling over nodes
            return F.log_softmax(h, dim=1)
        else:
            h = self.feat_dp2(h)
            h = torch.max(h, dim=1)[0].squeeze()  # max pooling over nodes
            h = self.classify(h)
            return F.log_softmax(h, dim=1)


