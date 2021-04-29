import os
import sys
sys.path.append(os.getcwd()+'/NGCF')
from dataset import DataTest, DataTrain
from model import NGCF
from torch.utils.data import DataLoader
from utility import *
import torch
import scipy.sparse as sp
import numpy as np

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    config = {
    'batch_size' : 1024,
    'path' : './Librarything',
    'neg_item' : 1,
    'n_embedding' : 64,
    'topk' : 10,
    'regularization' : 0.00001,
    'lr' : 0.001,
    'hgnr': False,
    'epoch' : 3000,
    'save_dir' : './NGCF/NGCF_Librarything_save_model.model',
    'dropout' : 0.3,
    'use_pretrain':True,
    'save_pretrain': True,
    'test_epoch':50
    }
    train_data = DataTrain(config)
    train_data.load_data()
    
    model = NGCF(train_data.n_user, train_data.n_item, n_embedding=config['n_embedding'], dropout=config['dropout']).cuda()
    if config['use_pretrain']:
        model.load_state_dict(torch.load(config['save_dir']))
    test_data = DataTest(config)
    test_data.load_test(train_data.n_item)

    test_dataloader = DataLoader(test_data, batch_size = config['batch_size']*2, shuffle=False) 
    optim = torch.optim.Adam(model.parameters(), lr = config['lr'])
    max_val = {'hit':0.0, 'ndcg': 0.0, 'h_epoch':0, 'ndcg_epoch':0}
    for e in range(config['epoch']):
        start = timer()
        train_data.make_batch_sampling()
        dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
        total_loss = 0.0
        
        for user, pos, neg in dataloader:
            model.train()
            optim.zero_grad()
            user = torch.LongTensor(user).cuda()
            pos_item = torch.LongTensor(pos).cuda()
            neg_item = torch.LongTensor(neg).cuda()

            user_embedding, item_embedding = model(train_data.L_c)
            user_batch_embed = user_embedding[user]
            p_i_batch_embed = item_embedding[pos_item]
            n_i_batch_embed = item_embedding[neg_item]
            bpr_loss, reg_loss = bpr(config['batch_size'], user_batch_embed, p_i_batch_embed, n_i_batch_embed, config['regularization'])
            loss = bpr_loss + reg_loss
            loss.backward()
            optim.step()
            total_loss += loss.item()
        if e % 10 == 0:
            print("Epoch : {:d}, Loss : {:4f}, Time : {:4f}".format(e, total_loss, timer()-start))
        if config['test_epoch'] <= e:
            
            test_timer = timer()
            model.eval()
            hit, ndcg = test(model, train_data, test_dataloader)
            if max_val['hit']<hit:
                max_val['hit'] = hit
                max_val['h_epoch'] = e
            if max_val['ndcg']<ndcg:
                max_val['ndcg'] = ndcg
                max_val['ndcg_epoch'] = e
            if e % 10 == 0:
                print("Epoch : {:d}, Hit@{:d} : {:4f},NDCG@{:d} : {:4f} MAX HIT :{:4f},MAX NDCG : {:4f}, Max at epoch : {:d} ,Time : {:4f}".format(e, config['topk'],hit, config['topk'], ndcg,max_val['hit'],max_val['ndcg'],max_val['h_epoch'],timer()-test_timer))
        if config['save_pretrain']:
            torch.save(model.state_dict(), config['save_dir'])