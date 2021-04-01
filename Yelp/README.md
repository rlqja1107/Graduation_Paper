# Train, Test 구성 방법  
* 한 User가 남긴 Review data(rating포함)를 모두 모은 후, 그 중에서 Random하게 0.8의 비율로 선택하고, 나머지 0.2를 test set으로 구성한다.  
* Code   
`review_reconstruct`의 경우는 (user, item ,rating)의 정보가 담겨있는 numpy 행렬    
`map_user_dict`의 경우는 Key는 user의 original id, value는 indexing한 번호의 dictinoary  
``` python  
interaction = []
train_set = {}
test_set = {}
exclude_case = {}
start = timer()
cold_start_user = 0
regular_start_user = 0
total_item = set()
map_dict = sorted(map_user_dict.values())
map_dict = torch.LongTensor(map_dict).cuda()
for user_id in map_dict:
    match = review_reconstruct[review_reconstruct[:,0] == user_id]
    match = match[:,1]
    user_id = user_id.item()
    if match.shape[0]>1:
        match = list(match.cpu().numpy())
        if len(match) >=10:
            regular_start_user+=1
        else:
            cold_start_user+=1
        train_length = floor(len(match)*0.8)
        train_set[user_id] = random.sample(match, train_length)
        test_case = set(match) - set(train_set[user_id])
        test_set[user_id] = list(test_case)
        n_train += len(train_set[user_id])
        n_test += len(test_set[user_id])
        total_item.update(match)
    else:
        exclude_case[user_id] = match
```    
시간 : 약 600초  
