# 졸업논문  

<h2 style="color: royalblue; font-weight: bold">주제</h2>  

Recommendation System에서 근본적인 문제 중 하나는 **Cold Start** Problem이다. Cold Start는 새로운 User 또는 Item이 추천시스템에 도입될 때 기본적인 Profile과 Interaction정보가 부족하여 새로운 User에게 추천을 하거나 새로운 Item을 추천하기가 어렵다.  

Cold Start를 추천시스템과 Graph를 접목시켜 어느정도 문제를 완화하고자 한다.  

<h2 style="color: royalblue; font-weight: bold">Target Paper</h2>  

* [A Heterogeneous Graph Neural Model for Cold-start Recommendation](https://dl.acm.org/doi/abs/10.1145/3397271.3401252)  
* [Neural Graph Collaborative Filtering](https://arxiv.org/pdf/1905.08108.pdf)  

<h2 style="color: royalblue; font-weight: bold">데이터셋</h2>  

* Librarything  
[Click](https://cseweb.ucsd.edu/~jmcauley/datasets.html#social_data)  
* Epinion Review   
[Click](https://cseweb.ucsd.edu/~jmcauley/datasets.html#social_data)   
* Yelp-2018   
[Click](https://www.kaggle.com/yelp-dataset/yelp-dataset)   

Train과 Test Set을 구성한 방법에서는 [여기](https://github.com/rlqja1107/Graduation_Paper/blob/main/Yelp/README.md)를 참조  

<h3 style="color: red; font-weight: bold">Epinion</h3>   

|Model|NDCG(구현)|HR(구현)||NCDG(논문)|HR(논문)|    
|:---:|:---:|:---:|:---:|:---:|:---:|      
|BPR|0.00046|0.001234||0.00606|0.00672|    
|NeuMF|0.045|0.0976||0.00841|0.00739|  
|NGCF|0.003655|0.006970||0.00850|0.00955|   
|HGNR|0.0.003231|0.001430||0.00945|0.00955|   

* HGNR: Learing Rate : 0.00005, Regularization : 0.005, Epoch : 1000(360에서 최대)

<h3 style="color: red; font-weight: bold">Librarything</h3>   

|Model|NDCG(구현)|HR(구현)||NCDG(논문)|HR(논문)|    
|:---:|:---:|:---:|:---:|:---:|:---:|      
|BPR||||||    
|NGCF|0.003045|0.002100||0.0801|0.0977|  
|HGNR||||||  

#### Dataset의 기본적인 설명보기   
[Click](https://github.com/rlqja1107/Graduation_Paper/wiki/Data-Explanation)

<h2 style="color: royalblue; font-weight: bold">이전모델의 성능비교</h2>  

