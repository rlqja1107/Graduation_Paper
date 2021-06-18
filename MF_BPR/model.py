import numpy as np
from tqdm import trange
from itertools import islice
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

class BPR:
    def __init__(self, learning_rate = 0.01, n_factors = 15, n_iters = 10,
                 batch_size = 1000, reg = 0.01, seed = 1234, verbose = True):
        self.reg = reg
        self.seed = seed
        self.verbose = verbose
        self.n_iters = n_iters
        self.n_factors = n_factors
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # to avoid re-computation
        self.prediction = None

    def fit(self, ratings):
        indptr = ratings.indptr #value counts in each row
        indices = ratings.indices
        n_users, n_items = ratings.shape

        # batch size should be smaller than number of users
        batch_size = self.batch_size
        if n_users < batch_size:
            batch_size = n_users

        batch_iters = n_users // batch_size

        # initialize random weights
        rstate = np.random.RandomState(self.seed)
        self.user_factors = rstate.normal(size = (n_users, self.n_factors))
        self.item_factors = rstate.normal(size = (n_items, self.n_factors))

        # progress bar
        loop = range(self.n_iters)
        if self.verbose:
            loop = trange(self.n_iters, desc = self.__class__.__name__)
        
        for _ in loop:
            for _ in range(batch_iters):
                sampled = self.sample(n_users, n_items, indices, indptr)
                sampled_users, sampled_pos_items, sampled_neg_items = sampled
                self.update(sampled_users, sampled_pos_items, sampled_neg_items)

        return self


    def sample(self, n_users, n_items, indices, indptr):
        """
        sample batches of random triplets u, i, j
        """
        sampled_pos_items = np.zeros(self.batch_size, dtype=np.int)
        sampled_neg_items = np.zeros(self.batch_size, dtype=np.int)
        sampled_users = np.random.choice(n_users, size = self.batch_size, replace = False)

        for idx, user in enumerate(sampled_users):
            pos_items = indices[indptr[user]:indptr[user+1]]
            pos_item = np.random.choice(pos_items)
            neg_item = np.random.choice(n_items)

            while neg_item in pos_items:
                neg_item = np.random.choice(n_items)

            sampled_pos_items[idx] = pos_item
            sampled_neg_items[idx] = neg_item

        return sampled_users, sampled_pos_items, sampled_neg_items
    
    def update(self, u, i, j):
        """
        update according to the bootstrapped user u,
        positive item i and negative item j
        """
        user_u = self.user_factors[u]
        item_i = self.item_factors[i]
        item_j = self.item_factors[j]

        
        r_uij = np.sum(user_u * (item_i - item_j), axis = 1)
        sigmoid = np.exp(-r_uij) / (1.0 + np.exp(-r_uij))

        # dimenstion matching
        sigmoid_tiled = np.tile(sigmoid, (self.n_factors, 1)).T

        # update using gradient descent
        grad_u = sigmoid_tiled * (item_j - item_i) + self.reg * user_u
        grad_i = sigmoid_tiled * -user_u + self.reg * item_i
        grad_j = sigmoid_tiled * user_u + self.reg * item_j
        self.user_factors[u] -= self.learning_rate * grad_u
        self.item_factors[i] -= self.learning_rate * grad_i
        self.item_factors[j] -= self.learning_rate * grad_j

        return self

    def predict(self):
        """
        Obtain the predicted ratingfs for every users and items
        """        
        
        if self.prediction is None:
            self.prediction = self.user_factors.dot(self.item_factors.T)

        return self.prediction

    def predict_user(self, user):
        """
        Returns the predicted ratings for the specified user
        """
        user_pred = self.user_factors[user].dot(self.item_factors.T)
        
        return user_pred

    def recommend(self, ratings, N = 5):
        """
        Returns the top N ranked items for given user id
        """

        n_users = ratings.shape[0]
        recommendation = np.zeros((n_users, N), dtype = np.uint32)
        for user in range(n_users):
            top_n = self.recommend_user(ratings, user, N)
            recommendation[user] = top_n

        return recommendation

    def recommend_user(self, ratings, user, N):
        """
        Top-N ranked items for specific user
        """
        scores = self.predict_user(user)

        liked = set(ratings[user].indices)
        count = N + len(liked)
        if count < scores.shape[0]:
            ids = np.argpartition(scores, -count)[-count:]
            best_ids = np.argsort(scores[ids])[::-1] 
            best = ids[best_ids]

        else:
            best = np.argsort(scores)[::-1]

        top_n = list(islice((rec for rec in best if rec not in liked), N))

        return top_n
    
    def get_similar_items(self, N=5, item_ids=None):
        """
        return the top N similar items for itemid by cosine distance
        """
        normed_factors = normalize(self.item_factors)
        knn = NearestNeighbors(n_neighbors = N + 1, metric = 'euclidean')
        knn.fit(normed_factors)

        if item_ids is not None:
            normed_factors = normed_factors[item_ids]

        _, items = knn.kneighbors(normed_factors)
        similar_items = items[:, 1:].astype(np.uint32)

        return similar_items

