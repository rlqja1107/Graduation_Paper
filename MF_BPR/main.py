from dataset import load_data
from model import BPR
from metric import auc_score, getHitRatio, getNDCG
from multiprocessing import Pool, Manager
import time
import sys

dataset = ['epinion82', 'epinion91', 'librarything82', 'librarything91'][3]
X_train, X_test, num_users_test, items_list_test, users_list_test = load_data(dataset)

n_iters_list = [10, 50, 100, 200, 500, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
lr_list = [1e-2, 1e-3, 1e-4]
batch_list = [64, 1024]

def bpr_returns(epoch):
    
    bpr_params = {'learning_rate': lr,
                  'batch_size': batch,
                  'n_iters': epoch,
                  'n_factors': 64,
                  'reg': 1e-4}

    bpr = BPR(**bpr_params)
    bpr.fit(X_train)
    
    start = time.time()
    top10 = bpr.recommend(X_train, N = 10)
    print('Time for recommend:', time.time() - start)

    top10_item = top10[users_list_test,:]

    top10_test = {}

    for i in range(len(users_list_test)):
        user_index = users_list_test[i]
        top10_test[user_index] = list(top10_item[i])

    HR_dict[epoch] = top10_test

    ndcg_list_top10 = []
    for i in range(num_users_test):
        ndcg_list_top10.append(getNDCG(top10[i], items_list_test[i], 10))

    hit_list_top10 = []
    for i in range(num_users_test):
        hit_list_top10.append(getHitRatio(top10[i], items_list_test[i]))

    print('------------')
    print('Epoch:', epoch)
    print('ndcg@10:', sum(ndcg_list_top10) / num_users_test)
    print('hit@10:', sum(hit_list_top10) / num_users_test)
    HR_list.append(sum(hit_list_top10) / num_users_test)
    print()

HR_best = {}
models = {}

for i in range(len(lr_list)):
    lr = lr_list[i]

    for j in range(len(batch_list)):
        batch = batch_list[j]

        with Manager() as manager:
            HR_list = manager.list()
            HR_dict = manager.dict()

            sys.stdout = open('./MF_BPR/results/' + dataset + '/lr_' + str(lr) + '_batch_' + str(batch) +'.txt', "w")

            start_time = time.time()
            pool = Pool(processes=20)
            pool.map(bpr_returns, n_iters_list)
            pool.close()
            pool.join()

            print('Done in:', time.time() - start_time, 'sec')
            sys.stdout.close()
 
            epoch_index = HR_list.index(max(HR_list))
            best_epoch = n_iters_list[epoch_index]
            best_model_top10 = HR_dict[best_epoch]

            HR_best[max(HR_list)] = (lr, batch, best_epoch)
            models[(lr, batch, best_epoch)] = best_model_top10

optimal_setting = HR_best[max(HR_best.keys())]
best_model = models[optimal_setting]

with open('./result/' + dataset + '/BPR_dataset_' + dataset + '_lr_' + str(optimal_setting[0]) + '_batch_size_' + str(optimal_setting[1]) + '_epoch_' + str(optimal_setting[2]) +'.txt', "w") as f:
    for k, v in best_model.items():
        string = ""
        string += str(k)
        for i in v:
            string += " "
            string += str(i)
        string += '\n'
        f.write(string)
