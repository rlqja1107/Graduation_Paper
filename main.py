from Data import EpinionData, EpinionTest
from torch.utils.data import DataLoader
import torch  
from Model import NGCF
from utility import bpr, test
from timeit import default_timer as timer
import multiprocessing

if __name__ == '__main__':
    config = {
        'batch_size' : 1024,
        'path' : './Epinion',
        'neg_item' : 1,
        'n_embedding' : 64,
        'topk' : 10,
        'core' : 8
    }
    pool = multiprocessing.Pool(config['core'])
    train_data = EpinionData(config)
    train_data.load_data()
    model = NGCF(train_data.n_user, train_data.n_item, n_embedding=config['n_embedding'])
    test_data = EpinionTest(config)
    test_dataloader = DataLoader(test_data, batch_size = config['batch_size']*2, shuffle=False) 
    optim = torch.optim.Adam(model.parameters(), lr = 0.01)
    for e in range(100):
        train_data.make_batch_sampling()
        dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
        total_loss = 0.0
        start = timer()
        for index, batch in enumerate(dataloader):
            user = torch.FloatTensor(batch[0])
            pos_item = torch.FloatTensor(batch[1])
            neg_item = torch.FloatTensor(batch[2])
            model.train()
            optim.zero_grad()
            user_embedding, item_embedding = model(data.H)
            user_batch_embed = user_embedding[user]
            p_i_batch_embed = item_embedding[pos_item]
            n_i_batch_embed = item_embedding[neg_item]
            loss = bpr(config['n_batch'], user_batch_embed, p_i_batch_embed, n_i_batch_embed)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        if e % 10 == 0:
            print("Epoch : {:d}, Loss : {:4f}, Time : {:4f}".format(e, total_loss, timer()-start))
        test_timer = timer()
        hit = test(model, train_data, test_dataloader, pool)
        if e % 10 == 0:
            print("Epoch : {:d}, Hit@K : {:4f}, Time : {:4f}".format(e, hit, timer()-test_timer))
            
