import os
import sys
sys.path.append(os.getcwd()+'/GP/HGNR')
from dataset import DataTest, DataTrain
from model import NGCF
from torch.utils.data import DataLoader
from utility import *
import torch
import scipy.sparse as sp
import numpy as np

def save_result(path, topk_result):
    with open(path, 'w') as f:
        for k, v in topk_result.items():
            l = [k] + list(v)
            f.write(" ".join(map(str, l)))
            f.write("\n")

if __name__ == '__main__':
    data_path = './dataset/'
    model_name = 'NGCF'
    
    config = {
    'gpu_num': '0',
    'batch_size' : [1024],
    'data_path' : [data_path+'epinion91', data_path+'epinion82', data_path+'librarything91', data_path+ 'librarything82'],
    'neg_item' : 1,
    'n_embedding' : 64,
    'topk' : 10,
    'regularization' : 0.00001,
    'lr' : [0.001, 0.0001, 0.00001],
    'hgnr': True,
    'epoch' : 2000,
    'save_model' : './HGNR/save_model/',
    'dropout' : 0.3,
    'save_pretrain': True,
    'test_epoch':10,
    'print_train_term': 100,
    'save_result': './result'
    }
    config['hgnr'] = True if model_name == 'HGNR' else False
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" if model_name == 'HGNR' else "0"
    
        
    for i, path in enumerate(config['data_path'][2:]):
        config['save_model'] = './HGNR/save_model/' + "{}_{}_".format(
                    'HGNR' if model_name == 'HGNR' else 'NGCF',
                    path[len(data_path):],
                    )

        train_data = DataTrain(config, path)
        train_data.load_data()
        test_data = DataTest(path)
        test_data.load_test(train_data.n_item)
        
        best_metric = {'hit': 0.0, 'ndcg': 0.0}
        best_topk_result = {'hit': {}, 'ndcg': {}}
        best_parameter = {'hit':{'lr':0.0, 'batch_size': 0.0}, 'ndcg':{'lr':0.0, 'batch_size': 0.0}}
        
        for lr in config['lr']:
            for batch_size in config['batch_size']:
            

                test_dataloader = DataLoader(test_data, batch_size = batch_size*2, shuffle=False)
                
                model = NGCF(train_data.n_user, train_data.n_item, n_embedding=config['n_embedding'], dropout=config['dropout']).cuda()
                optim = torch.optim.Adam(model.parameters(), lr = lr)
                
                
                max_val = {'hit': 0, 'ndcg': 0, 'h_epoch': 0}
                for e in range(config['epoch']):
                    start = timer()
                    train_data.make_batch_sampling()
                    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
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
                        bpr_loss, reg_loss = bpr(batch_size, user_batch_embed, p_i_batch_embed, n_i_batch_embed, config['regularization'])
                        loss = bpr_loss + reg_loss
                        loss.backward()
                        optim.step()
                        total_loss += loss.item()
                    # For print
                    if e % config['print_train_term'] == 0:
                        print("Epoch : {:d}, Loss : {:4f}, Time : {:4f}".format(e, total_loss, timer()-start))

                    if e% 150 == 0:
                        print("Model : {}, Data: {},lr: {}, batch: {}, Loss: {:4f}, Train Time : {:4f}".format(model_name, 
                        'Library91' if i in [0] else "Library82",
                        lr,batch_size, total_loss, timer()-start))

                    if config['test_epoch'] <= e:          
                        test_timer = timer()
                        model.eval()
                        hit, ndcg, topk_result = test(model, train_data, test_dataloader)
                        if max_val['hit']<hit:
                            max_val['hit'] = hit
                            max_val['h_epoch'] = e
                            print("Epoch : {:d}, Max Hit: {:4f}, Max_NDCG: {:4f}, Test Time : {:4f}".format(e, max_val['hit'], max_val['ndcg'], timer()-test_timer))
                            if best_metric['hit'] < hit:
                                best_metric['hit'] = hit
                                torch.save(model.state_dict(), config['save_model']+"Max_Hit.model")
                                best_topk_result['hit'] = topk_result
                                best_parameter['hit']['lr'] = lr
                                best_parameter['hit']['batch_size'] = batch_size
                            
                        if max_val['ndcg']<ndcg:
                            max_val['ndcg'] = ndcg
                            if best_metric['ndcg'] < ndcg:
                                best_metric['ndcg'] = ndcg
                                torch.save(model.state_dict(), config['save_model']+"Max_Ndcg.model")
                                best_topk_result['ndcg'] = topk_result
                                best_parameter['ndcg']['lr'] = lr
                                best_parameter['ndcg']['batch_size'] = batch_size
                                
                                
        save_result(path='./result/{}/{}_{}_lr{}_batch{}_ndcg_result.txt'.format(
                                path[len(data_path):],
                                'HGNR' if model_name == 'HGNR' else 'NGCF',
                                'Epinion' if i in [0,1] else "Library",
                                best_parameter['ndcg']['lr'],
                                best_parameter['ndcg']['batch_size']
                                ), topk_result=best_topk_result['ndcg'])

        save_result(path='./result/{}/{}_{}_lr{}_batch{}_hit_result.txt'.format(
                                path[len(data_path):],
                                'HGNR' if model_name == 'HGNR' else 'NGCF',
                                'Epinion' if i in [0,1] else "Library",
                                best_parameter['hit']['lr'],
                                best_parameter['hit']['batch_size']
                                ), topk_result=best_topk_result['hit'])
                        
                        