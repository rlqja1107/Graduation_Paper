import json
import pandas as pd
import numpy as np
from ast import literal_eval

cut = 91  #82 or 91
origin_data_path = "../origin_dataset/epinion/"
preprocessing_path = "../preprocessing_data/epinion{}".format(cut)
dataset_path = "../dataset/epinion{}".format(cut)

#
print("--- loading dataset ----")
data = []
with open(origin_data_path+"/epinions.json", "r") as f:
    for line in f:
        line = json.dumps(line)
        data.append(json.loads(line))
_dict = {'user': [], 'stars': [], 'time': [], 'paid': [], 'item': [], 'review': []}
for row in data:
    row = row.replace("\n", "")
    row_dict = literal_eval(row)
    for k, v in row_dict.items():
        _dict[k].append(v)
df = pd.DataFrame(_dict)
print("--- finish load dataset ---")

#
df = df.drop_duplicates(subset=['user', 'stars', 'time', 'item'])
#
count_lower_bound = 2
user_grouped = df.groupby(df['user'])
rating_count_df = pd.DataFrame(user_grouped['stars'].count()).reset_index()
rating_count_df.rename(columns={'user': 'user', 'stars': 'count'}, inplace=True)
user_rating_df = pd.merge(df, rating_count_df, left_on='user', right_on='user', how='inner')
user_rating_df = user_rating_df[user_rating_df['count'] >= count_lower_bound]
#
print("--- split dataset ---")
time_list = sorted(user_rating_df['time'])
cutting_dict = {82: 0.5, 91: 0.75}
time_cut = time_list[int(len(time_list)*cutting_dict[cut])]


def time_cutting(df, time_cut):
    need_col = ['user', 'stars', 'time', 'item', 'review', 'count']
    train = df[df['time'] <= time_cut][need_col]
    test = df[df['time'] > time_cut][need_col]

    test_list = []
    test_val = test.values.tolist()
    train_user = {k: 0 for k in set(train['user'])}
    for row in test_val:
        user = row[0]
        if user in train_user:
            test_list.append(row)
    test = pd.DataFrame(test_list, columns=need_col)
    print("train: {}".format(round(train.shape[0]/(train.shape[0]+test.shape[0]), 3)))
    print("train: {}".format(round(test.shape[0]/(train.shape[0]+test.shape[0]), 3)))
    return train, test
train, test = time_cutting(user_rating_df, time_cut)

#
item_to_index = {}
num_item = 0
user_to_index = {}
num_user = 0


def user_change_index_func(x):
    global num_user
    if x in user_to_index:
        return user_to_index[x]
    else:
        user_to_index[x] = num_user
        num_user += 1
        return user_to_index[x]


def item_change_index_func(x):
    global num_item
    if x in item_to_index:
        return item_to_index[x]
    else:
        item_to_index[x] = num_item
        num_item += 1
        return item_to_index[x]


encoding_train_df = train.copy()
encoding_test_df = test.copy()
encoding_train_df['user'] = train['user'].apply(user_change_index_func)
encoding_train_df['item'] = train['item'].apply(item_change_index_func)
encoding_test_df['user'] = test['user'].apply(user_change_index_func)
encoding_test_df['item'] = test['item'].apply(item_change_index_func)
print("---make! train, test ---")

#
trust = pd.read_csv(origin_data_path + "/network_trust.txt", sep=" ", header=None, names=['give', 'trust', 'take'])
trusted = pd.read_csv(origin_data_path + "/network_trustedby.txt", sep=" ", header=None, names=['take', 'trusted', 'give'])
df = pd.merge(trust, trusted, left_on=['give', 'take'], right_on=['give', 'take'], how='outer')
rating_user = num_user
df['give'] = df['give'].apply(user_change_index_func)
df['take'] = df['take'].apply(user_change_index_func)
network_df = df[(df['give'] < rating_user) & (df['take'] < rating_user)]

#
cold_bound = 10
train_interaction = encoding_train_df.shape[0]
test_interaction = encoding_test_df.shape[0]
num_cold_user = len(set(encoding_train_df[encoding_train_df['count'] < cold_bound]['user']))
num_regular_user = len(set(encoding_train_df[encoding_train_df['count'] >= cold_bound]['user']))
num_social_edge = network_df.shape[0]

print("--- info ---")
print("user: {}".format(rating_user))
print("item: {}".format(num_item))
print("train_interaction: {}".format(train_interaction))
print("test_interaction: {}".format(test_interaction))
print("train_cold_user: {}".format(num_cold_user))
print("train_regular_user: {}".format(num_regular_user))
print("social_edge: {}".format(num_social_edge))

#
sorted_train_df = encoding_train_df.sort_values(by=['user', 'time'])
sorted_test_df = encoding_test_df.sort_values(by=['user', 'time'])
sorted_train_item_review_df = encoding_train_df.sort_values(by=['item'])
sorted_network_df = network_df.sort_values(by=['give', 'take'])

need_r = ['user', 'item']
need_s = ['give', 'take']
need_c = ['item', 'review']
train_r_df = sorted_train_df[need_r]
test_r_df = sorted_test_df[need_r]
s_df = sorted_network_df[need_s]
c_df = sorted_train_item_review_df[need_c]

# save - R
train_r_list = train_r_df.values.tolist()
test_r_list = test_r_df.values.tolist()
train_dict =  {}
test_dict = {}

for row in train_r_list:
    user, item = row
    if user in train_dict:
        train_dict[user].append(item)
    else:
        train_dict[user] = [item]
for row in test_r_list:
    user, item = row
    if user in test_dict:
        test_dict[user].append(item)
    else:
        test_dict[user] = [item]



with open(dataset_path+"/train.txt", "w") as f:
    for k, v in train_dict.items():
        string = ""
        string += str(k)
        for i in v:
            string += " "
            string += str(i)
        string += "\n"
        f.write(string)

with open(dataset_path+"/test.txt", "w") as f:
    for k, v in test_dict.items():
        string = ""
        string += str(k)
        for i in v:
            string += " "
            string += str(i)
        string += "\n"
        f.write(string)


# save - S
s_list = s_df.values.tolist()
s_dict = {}
for row in s_list:
    give, take = row
    if give in s_dict:
        s_dict[give].append(take)
    else:
        s_dict[give] = [take]

with open(preprocessing_path + "/network.txt", "w") as f:
    for k, v in s_dict.items():
        string = ""
        string += str(k)
        for i in v:
            string += " "
            string += str(i)
        string += "\n"
        f.write(string)


# save - C
c_list = c_df.values.tolist()
c_dict = {}
for row in c_list:
    item, review = row
    if item in c_dict:
        c_dict[item].append(review)
    else:
        c_dict[item] = [review]


with open(preprocessing_path+"/review.txt", "w") as f:
    for k, v in c_dict.items():
        string = ""
        string += str(k)
        for i in v:
            string += " "
            string += i
        string += "\n"
        f.write(string)
