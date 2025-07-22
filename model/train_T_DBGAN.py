import os
import math
import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx

from utils import * 
from metrics import * 
import pickle
import argparse
from torch import autograd
import torch.optim.lr_scheduler as lr_scheduler
from T_DBGAN import *
from att_model import *


parser = argparse.ArgumentParser()

#Model specific parameters
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_fecnn', type=int, default=2,help='Number of fecnn layers')
parser.add_argument('--n_tpcnn', type=int, default=4, help='Number of TPCNN layers')
parser.add_argument('--kernel_size', type=int, default=3)

#Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)

parser.add_argument('--dataset_eth', default='eth', help='eth,hotel,univ,zara1,zara2')  
parser.add_argument('--dataset_hotel', default='hotel', help='eth,hotel,univ,zara1,zara2')  
parser.add_argument('--dataset_univ', default='univ', help='eth,hotel,univ,zara1,zara2')  
parser.add_argument('--dataset_zara1', default='zara1', help='eth,hotel,univ,zara1,zara2')  
parser.add_argument('--dataset_zara2', default='zara2', help='eth,hotel,univ,zara1,zara2')  

#Training specifc parameters
parser.add_argument('--batch_size', type=int, default=128,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=250,
                    help='number of epochs')  
parser.add_argument('--clip_grad', type=float, default=None,
                    help='gadient clipping')        
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=150,
                    help='number of steps to drop the lr')  
parser.add_argument('--use_lrschd', action="store_true", default=False,
                    help='Use lr rate scheduler')
parser.add_argument('--tag', default='tag',
                    help='personal tag for the model ')
parser.add_argument('--optim_type', type=str, default='SGD', help='optimizer type')
parser.add_argument('--model_type', type=str, default='att_res_gcnn', help='model')
                    
args = parser.parse_args()


print('*'*30)
print("Training initiating....")
print(args)


def graph_loss(V_pred,V_target):
    return bivariate_loss(V_pred,V_target)

#Data prep     
obs_seq_len = args.obs_seq_len
pred_seq_len = args.pred_seq_len

dataset_eth = './datasets/'+args.dataset_eth+'/'

# train_loader

dset_train = TrajectoryDataset(
            dataset_eth+'train/',
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            skip=1,norm_lap_matr=False)

dset_val = TrajectoryDataset(
            dataset_eth+'val/',
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            skip=1,norm_lap_matr=False)


with open('./dbgan_embd/emb_lst.pkl', 'rb') as f:
    data_dict = pickle.load(f)

with open('./dbgan_embd_val/emb_lst.pkl', 'rb') as f:
    val_data_dict = pickle.load(f)

data_lst = data_dict['emb']
data_lst_val = val_data_dict['emb']  

# Define Dataset for Loader
class Mydatset(Dataset):
    def __init__(self, emb_list, obs_list, pred_list, A_obs_lst, A_tr_lst):
        super().__init__()

        self.emb_list = emb_list
        self.obs_list = obs_list
        self.pred_list = pred_list
        self.A_obs_lst = A_obs_lst
        self.A_tr_lst = A_tr_lst
    
    def __len__(self):
        return len(self.obs_list)
    
    def __getitem__(self, index):
        return self.emb_list[index], self.obs_list[index], self.pred_list[index], self.A_obs_lst[index], self.A_tr_lst[index]

# train
train_loader = DataLoader(dset_train, batch_size=1, shuffle=False)

train_data_lst = []
train_data_pred_lst = []
A_obs_lst = []
A_tr_lst = []
for batch in train_loader:
    v_obs = batch[-4]
    v_tr = batch[-2]
    A_obs = batch[-3]
    A_tr = batch[-1]
    train_data_lst.append(v_obs.squeeze())
    train_data_pred_lst.append(v_tr.squeeze())
    A_obs_lst.append(A_obs.squeeze())
    A_tr_lst.append(A_tr.squeeze())


embd_data_lst = []
for i in range(len(data_lst)):
    embd_data_lst.append(data_lst[i][0]) # obs_len, N, 2


train_datasets = Mydatset(embd_data_lst, train_data_lst, train_data_pred_lst, A_obs_lst, A_tr_lst)

loader_train = DataLoader(train_datasets, batch_size=1, shuffle=True, num_workers=0)

# val
val_loader = DataLoader(dset_val, batch_size=1, shuffle=False)
val_data_lst = []
val_data_pred_lst = []
A_obs_lst_val = []
A_tr_lst_val = []
for batch in val_loader:
    v_obs = batch[-4]
    v_tr = batch[-2]
    A_obs = batch[-3]
    A_tr = batch[-1]
    val_data_lst.append(v_obs.squeeze())
    val_data_pred_lst.append(v_tr.squeeze())
    A_obs_lst_val.append(A_obs.squeeze())
    A_tr_lst_val.append(A_tr.squeeze())

embd_data_val_lst = []
for i in range(len(data_lst_val)):
    embd_data_val_lst.append(data_lst_val[i][0])

val_datasets = Mydatset(embd_data_val_lst, val_data_lst, val_data_pred_lst, A_obs_lst_val, A_tr_lst_val)
loader_val = DataLoader(val_datasets, batch_size=1, shuffle=False, num_workers=0)


# model 
if args.model_type == 'res_gnn':
    model = res_gcnn(n_fecnn = args.n_fecnn, n_tpcnn=args.n_tpcnn ,
                    output_feat=args.output_size,seq_len=args.obs_seq_len,
                    kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).cuda()

if args.model_type == 'att_res_gcnn':
    model = res_gcnn(n_fecnn = args.n_fecnn, n_tpcnn=args.n_tpcnn ,
                    output_feat=args.output_size,seq_len=args.obs_seq_len,
                    kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).cuda()



#Training settings
if args.optim_type == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

if args.optim_type == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

if args.use_lrschd:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)
    


checkpoint_dir = './checkpoint_dbgan/'+args.tag+'/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
with open(checkpoint_dir+'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)
    

print('Data and model loaded')
print('Checkpoint dir:', checkpoint_dir)

#Training 
metrics = {'train_loss':[],  'val_loss':[]}
constant_metrics = {'min_val_epoch':-1, 'min_val_loss':9999999999999999}


def train(epoch, loader_train):
    global metrics
    model.train()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point =int(loader_len/args.batch_size)*args.batch_size+ loader_len%args.batch_size -1

    for cnt,batch in enumerate(loader_train): 
        batch_count+=1

        #Get data
        batch = [tensor.cuda() for tensor in batch]
        emb, V_obs, V_tr, A_obs, A_tr = batch

        optimizer.zero_grad()
        #Forward
        #V_obs = batch,seq,node,feat
        #V_obs_tmp = batch,feat,seq,node
        V_obs_tmp =V_obs.permute(0,3,1,2)
        A_obs = A_obs.squeeze()

        V_pred = model(V_obs_tmp, emb)

        # v_pred -> batch_size, out_feat, pred_seq_len, N -> batch, pred_len, N, out_feat

        V_pred = V_pred.permute(0,2,3,1)
        
        V_tr = V_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count%args.batch_size !=0 and cnt != turn_point :
            l = graph_loss(V_pred,V_tr)
            if is_fst_loss :
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss/args.batch_size
            is_fst_loss = True
            loss.backward()
            
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)


            optimizer.step()
            #Metrics
            loss_batch += loss.item()
            print('TRAIN:','\t Epoch:', epoch,'\t Loss:',loss_batch/batch_count)
            
    metrics['train_loss'].append(loss_batch/batch_count)
    

def vald(epoch, loader_val):
    global metrics,constant_metrics
    model.eval()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    turn_point = int(loader_len/args.batch_size)*args.batch_size+ loader_len%args.batch_size -1
    
    for cnt,batch in enumerate(loader_val): 
        batch_count+=1

        #Get data
        batch = [tensor.cuda() for tensor in batch]
        emb, V_obs, V_tr, A_obs, A_tr = batch
        
        V_obs_tmp = V_obs.permute(0,3,1,2)
        A_obs = A_obs.squeeze()
        V_pred = model(V_obs_tmp, emb)
        
        V_pred = V_pred.permute(0,2,3,1)
        
        V_tr = V_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count%args.batch_size !=0 and cnt != turn_point :
            l = graph_loss(V_pred,V_tr)
            if is_fst_loss :
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss/args.batch_size
            is_fst_loss = True
            #Metrics
            loss_batch += loss.item()
            print('VALD:','\t Epoch:', epoch,'\t Loss:',loss_batch/batch_count)

    metrics['val_loss'].append(loss_batch/batch_count)
    
    if  metrics['val_loss'][-1]< constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] =  metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(),checkpoint_dir+'val_best.pth')  # OK


def train_gyw(loader_train, loader_val):
    print('Training started ...')
    for epoch in range(args.num_epochs):
        train(epoch, loader_train)
        vald(epoch, loader_val)
        if args.use_lrschd:
            scheduler.step()

        print('*'*30)
        print('Epoch:',args.tag,":", epoch)
        for k,v in metrics.items():
            if len(v)>0:
                print(k,v[-1])

        print(constant_metrics)
        print('*'*30)
        
        with open(checkpoint_dir+'metrics.pkl', 'wb') as fp:
            pickle.dump(metrics, fp)
        
        with open(checkpoint_dir+'constant_metrics.pkl', 'wb') as fp:
            pickle.dump(constant_metrics, fp)  


if __name__ == '__main__':
    train_gyw(loader_train, loader_val)
