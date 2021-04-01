# Train, Test 구성 방법  
* 한 User가 남긴 Review data(rating포함)를 모두 모은 후, 그 중에서 Random하게 0.8의 비율로 선택하고, 나머지 0.2를 test set으로 구성한다.  
* Code   
`review_reconstruct`의 경우는 (user, item ,rating)의 정보가 담겨있는 numpy 행렬    
`map_user_dict`의 경우는 Key는 user의 original id, value는 indexing한 번호의 dictinoary  
``` python  
start = timer()
train_set = {}
test_set = {}
for k,v in train_items.items():
    length = len(v)
    if length>=2:
        train_length = floor(length*0.8)
        train_set[k] = random.sample(v, train_length)
        test_case = set(v) - set(train_set[k])
        test_set[k] = list(test_case)
```    
* Yelp의 경우 Pytorch로 GPU를 사용했기 때문에 위와의 코드는 다르지만, Train과 Test Set을 구성하는 방법에서는 동일.  
시간 : 약 600초  
