# 졸업논문  

<h2 style="color: royalblue; font-weight: bold">주제</h2>  

* Clustering

NGCF에서 User-User, Item-Item의 message passing을 다루기 위해서 HGNR Methodology가 나왔다. 하지만, HGNR에서는 User-User를 Friend에서 BPR를 이용해서 Top 20의 User끼리 이어주었다. 여기서 새롭게 User-User를 연결시켜주기 위해 agglomerative hierarchical clustering을 이용하여 Cluster에 있는 User끼리 연결시켜주었다. Item-Item에서는 HGNR에서는 Review를 S-BERT를 이용해서 Top 20의 Similarity가 높은 Item끼리 연결시켜주었다. 하지만, Item의 이름을 S-BERT로 Word Embedding을 하여 Top 20의 Item을 연결시켜 Message Passing을 시켜주었다.  

***  

* WSNG(Weight Sequence Learning For Neural Graph Collaborative Filtering)  

기존의 NGCF는 Static Graph로 시간의 정보를 반영하고 있지 않다. 시간의 정보를 반영하기 위해 EvolveGCN의 Idea를 활용하여 Dynamic Graph Representation을 하고자 한다. 새로운 모델의 이름을 **WSNG**으로 한다. 자세한 코드는 **WSNG** 디렉토리를 통해 확인할 수 있다.  

<h2 style="color: royalblue; font-weight: bold">Target Paper</h2>  

* [A Heterogeneous Graph Neural Model for Cold-start Recommendation](https://dl.acm.org/doi/abs/10.1145/3397271.3401252)  
* [Neural Graph Collaborative Filtering](https://arxiv.org/pdf/1905.08108.pdf)  
* [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/abs/1902.10191)  

<h2 style="color: royalblue; font-weight: bold">데이터셋</h2>  

* Librarything  
[Click](https://cseweb.ucsd.edu/~jmcauley/datasets.html#social_data)  
* Epinion Review   
[Click](https://cseweb.ucsd.edu/~jmcauley/datasets.html#social_data)   
* Yelp-2018   
[Click](https://www.kaggle.com/yelp-dataset/yelp-dataset)   

Train과 Test Set을 구성한 방법에서는 [여기](https://github.com/rlqja1107/Graduation_Paper/blob/main/Yelp/README.md)를 참조  

<h2 style="color: royalblue; font-weight: bold">데이터 결과</h2>

Time Sampling : Train과 Test를 나누는 기준을 한 TimeStamp를 기준으로 나눈다. 한 TimeStamp를 기준으로 이전 시점은 Train, 이후 시점은 Test로 이용하면서 Train 0.8, Test 0.2를 Epinion82라고 하고, Train 0.9, Test 0.1를 Epinion91이라고 한다.  

---  

<h3 style="color: red; font-weight: bold">Epinion82</h3>   

|Model|NDCG(구현)|HR(구현)|   
|:---:|:---:|:---:|   
|BPR|0.00094|0.000956|   
|NGCF|0.000389|0.004399|     
|HGNR|0.000505|0.004973|    
|HGNR_cluster|0.000329|0.005164|   
|WSNG|0.000666|0.006121|  

<h3 style="color: red; font-weight: bold">Epinion91</h3>   

|Model|NDCG(구현)|HR(구현)|  
|:---:|:---:|:---:|    
|BPR|0.000219|0.001513|     
|NGCF|0.000639|0.006305|    
|HGNR|0.000579|0.005801|     
|HGNR_cluster|0.000426|0.004035|    
|WSNG|0.000888|0.007062|  

<h3 style="color: red; font-weight: bold">Librarything - Time Sampling</h3>   

|Model|NDCG(구현)|HR(구현)|  
|:---:|:---:|:---:|     
|BPR|||  
|NGCF|0.012697|0.093611|  
|HGNR|0.011586|0.084920|   


> NDCG를 구하는 방법의 차이 

Target 논문에서는 NDCG에서 IDCG를 구할 때, hit list가 [0,1,0,0,....1]이라 할때 [1,1,0,...0]을 통해 구한다. 하지만 이 방법으로는 정확한 비교가 불가능하다고 판단했다. 따라서 NDCG에서 IDCG를 구하는데 있어, [1,1,0,...0]이 아니라 [1,1,1,....1]로 놓고 IDCG를 구하여 NDCG를 구했다. 따라서, 기존 논문보다 당연히 더 NDCG가 작아질 수 밖에 없다.  

### Dataset의 기본적인 설명보기   
[Click](https://github.com/rlqja1107/Graduation_Paper/wiki/Data-Explanation)  


