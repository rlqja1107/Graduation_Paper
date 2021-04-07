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
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    config = {
    'batch_size' : 1024,
    'path' : './Epinion',
    'neg_item' : 1,
    'n_embedding' : 64,
    'topk' : 10,
    'regularization' : 0.005,
    'lr' : 0.01
    }
    train_data = DataTrain(config)
    train_data.load_data()
    model = NGCF(train_data.n_user, train_data.n_item, n_embedding=config['n_embedding']).cuda()
    test_data = DataTest(config)
    test_data.load_test()
    test_dataloader = DataLoader(test_data, batch_size = config['batch_size']*2, shuffle=False) 
    optim = torch.optim.Adam(model.parameters(), lr = config['lr'])
    sparse_eye = sp.eye(train_data.n_user+train_data.n_item, dtype = np.float)
    sparse_eye = sparse_mx_to_torch_sparse_tensor(sparse_eye).cuda()

    for e in range(150):
        train_data.make_batch_sampling()
        dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
        total_loss = 0.0
        start = timer()
        for user, pos, neg in dataloader:
            model.train()
            optim.zero_grad()
            user = torch.LongTensor(user).cuda()
            pos_item = torch.LongTensor(pos).cuda()
            neg_item = torch.LongTensor(neg).cuda()

            user_embedding, item_embedding = model(train_data.L_c, sparse_eye)
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
        test_timer = timer()
        hit, ndcg = test(model, train_data, test_dataloader, sparse_eye)
        if e % 10 == 0:
            print("Epoch : {:d}, Hit@{:d} : {:4f},NDCG@{:d} : {:4f} Time : {:4f}".format(e, config['topk'],hit, config['topk'], ndcg,timer()-test_timer))
            
