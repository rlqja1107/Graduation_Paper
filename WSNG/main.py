import os
import sys
sys.path.append(os.getcwd()+'/NGCF_NEW')
from dataset import DataTest, DataTrain
from model import NGCF, LSTM
from torch.utils.data import DataLoader
from utility import *
import torch
import scipy.sparse as sp
import numpy as np
import pandas as pd
from timeit import default_timer as timer
import warnings
import pickle
from torch.optim.lr_scheduler import CosineAnnealingLR
warnings.filterwarnings(action='ignore')

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="2"

    config = {
    'batch_size' : [128,256,512],
    'path' : './Epinion',
    'neg_item' : 1,
    'n_embedding' : 64,
    'topk' : 10,
    'regularization' : 0.00001,
    'lr' : [0.001],
    'hgnr': False,
    'save_dir' :  './NGCF_NEW/ngcf_epinion_91_v2_lstm_sep.model',
    'dropout' : 0.3,
    'use_pretrain':False,
    'save_pretrain': False,
    'test': False,
    'total_range': [2000,],
    'max_count' : 5,
    'lstm_reg': [ 1e-07],
    'lstm_lr': [ 0.0001,0.00001,0.000001,5e-07]
    
    }
    if config['test']:
        train_data = DataTrain(config, 256)
        train_data.load_data()
        model = NGCF(train_data.n_user, train_data.n_item, n_embedding=64, dropout=config['dropout']).cuda()
        model.load_state_dict(torch.load(config['save_dir']))
        test_data = DataTest(config)
        test_data.load_test(train_data.n_item)
        test_dataloader = DataLoader(test_data, batch_size = 256*2, shuffle=False)
        i_range = torch.arange(0, train_data.n_item, dtype = torch.long).cuda()
        hit, ndcg = test(model, train_data.adj_list[len(train_data.adj_list)-1], test_dataloader, train_data, i_range)
        print("hit : {:4f}, ndcg : {:4f}".format(hit, ndcg))
        sys.exit()

    
    if config['use_pretrain']:
        model.load_state_dict(torch.load(config['save_dir']))


    
    max_info = {'range': 0, 'batch_size':0, 'lr':0, 'hit':0.0, 'ndcg': 0.0, 'Time': 0.0, 'lstm_lr': 0.0, 'lstm_reg':0.0}
    for batch_size in config['batch_size']:
        train_data = DataTrain(config, batch_size)
        train_data.load_data()
        
        test_data = DataTest(config)
        test_data.load_test(train_data.n_item)
        test_dataloader = DataLoader(test_data, batch_size = batch_size*2, shuffle=False)
     
        i_range = torch.arange(0, train_data.n_item, dtype = torch.long).cuda()
        for lr in config['lr']:
            for regula in config['lstm_reg']:
                for lstm_lr in config['lstm_lr']:
                    for r in config['total_range']:
                        print("r :{}, lr: {}, batch size :{}, lstm_lr: {}, lstm_reg: {}".format(r,lr,batch_size, lstm_lr, regula))
                        model = NGCF(train_data.n_user, train_data.n_item, n_embedding=64, dropout=config['dropout']).cuda()
                        optim = torch.optim.Adam(model.parameters(), lr = lr)
                        # scheduler = CosineAnnealingLR(optim, T_max=10)
                        lstm = LSTM(64).cuda()
                        # lstm_1_optim = torch.optim.Adam(lstm.parameters(), lr =lstm_lr)
                        lstm_1_optim = torch.optim.Adam(lstm.gru_1_layer.parameters(), lr =lstm_lr)
                        lstm_2_optim = torch.optim.Adam(lstm.gru_2_layer.parameters(), lr = lstm_lr)
                        lstm_3_optim = torch.optim.Adam(lstm.gru_3_layer.parameters(), lr = lstm_lr)
                        start = timer()
                        
                        for i, batch in enumerate(train_data.batch_list):
                            sequence_weight(model, lstm)
                            last_time = batch[len(batch)-1, 2]
                            df = train_data.train_sort[train_data.train_sort['time'] <= last_time][['user','item']].values.tolist()
                            n = len(df)
                            mapping = map(tuple,df)
                            one = np.zeros(n)
                            dic = dict(zip(mapping,one))
                            total_loss = 0.0
                            early_stop = {'cur_loss':10000000000.0, 'cur_count':0, 'max_count': config['max_count']}
                            early_stop_lstm = {'cur_loss':10000000000.0, 'cur_count':0, 'max_count': 5}
                            model.train()
                            for _ in range(r):
                                
                                neg_user = []
                                for u in batch[:,0]:
                                    neg_user += train_data.neg_sampling(u, dic)
                                
                                optim.zero_grad()
                                user = torch.LongTensor(batch[:,0]).cuda()
                                pos_item = torch.LongTensor(batch[:,1]).cuda()
                                neg_item = torch.LongTensor(neg_user).cuda()
                                user_embedding, item_embedding = model(train_data.adj_list[i]) 
                                user_batch_embed = user_embedding[user]
                                p_i_batch_embed = item_embedding[pos_item]
                                n_i_batch_embed = item_embedding[neg_item]
                                bpr_loss, reg_loss = bpr(batch_size, user_batch_embed, p_i_batch_embed, n_i_batch_embed, config['regularization'])
                                loss = bpr_loss + reg_loss
                                loss.backward()
                                optim.step()
                                # scheduler.step()
                                loss = loss.item()
                                if early_stop['cur_loss'] > loss:
                                    early_stop['cur_loss'] = loss
                                    early_stop['cur_count'] = 0
                                else:
                                    early_stop['cur_count'] += 1
                                    if early_stop['cur_count'] == early_stop['max_count']:
                                        break

                                total_loss += loss
                            model.eval()
                            for _ in range(r):
                                lstm.train()
                                loss_1, loss_2, loss_3, reg_1, reg_2, reg_3 = lstm_loss(model, lstm)
                                # loss_sum = sum(loss_1)+sum(loss_2)+sum(loss_3)
                                
                                lstm_1_optim.zero_grad()
                                # (regula*loss_sum).requires_grad_(True).backward()
                                (sum(loss_1)+regula*reg_1).requires_grad_(True).backward()
                                lstm_1_optim.step()

                                lstm_2_optim.zero_grad()
                                (sum(loss_2)+ regula*reg_2).requires_grad_(True).backward()
                                lstm_2_optim.step()

                                lstm_3_optim.zero_grad()
                                (sum(loss_3)+ regula*reg_3).requires_grad_(True).backward()
                                lstm_3_optim.step()
                                lstm.eval()
                                if early_stop_lstm['cur_loss'] > loss:
                                    early_stop_lstm['cur_loss'] = loss
                                    early_stop_lstm['cur_count'] = 0
                                else:
                                    early_stop_lstm['cur_count'] += 1
                                    if early_stop_lstm['cur_count'] == early_stop_lstm['max_count']:
                                        break
                            change_weight(model, lstm)
                        sequence_weight(model, lstm)
                        hit, ndcg = test(model, train_data.adj_list[len(train_data.adj_list)-1], test_dataloader, train_data, i_range)
                        print("Hit : {:4f}, NDCG : {:4f}".format(hit, ndcg))
                        if hit > max_info['hit']:
                            max_info['hit'] = hit
                            max_info['Time'] = timer()-start
                            max_info['range'] = r
                            max_info['batch_size'] = batch_size
                            max_info['lr'] = lr
                            max_info['ndcg'] = ndcg
                            max_info['lstm_lr'] = lstm_lr
                            max_info['lstm_reg'] = regula
                            print("Total Time : {:4f}, hit : {:4f}, ndcg : {:4f}, range : {:d}, batch_size : {:d}, lr:{}, lstm_lr:{}, lstm_reg:{}, max_count:{}".format(timer()-start, hit, ndcg,r,batch_size, lr,lstm_lr,regula,config['max_count']))
                            # torch.save(model.state_dict(), config['save_dir'])
    # with open("./NGCF_NEW/test_3_epinion91.pickle", 'wb') as f:
    #     pickle.dump(max_info, f)