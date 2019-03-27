import pandas as pd
import json
from tqdm import tqdm
import numpy as np


def read_json(path):
    with open(path, 'r') as f:
        output = json.load(f)
    return output


PATH_TEST = './data/test_users.json'
PATH_TRANSACTIONS = './data/transactions.csv'
PATH_ITEMS = './features/items_features.csv'
PATH_USERS = './features/users_features.csv'
PATH_RATINGS = './data/ratings.csv'
PATH_BOOKMARKS = './data/bookmarks.csv'

transactions = pd.read_csv(PATH_TRANSACTIONS)
users = pd.read_csv(PATH_USERS)
items = pd.read_csv(PATH_ITEMS)
test_users = read_json(PATH_TEST)['users']
ratings = pd.read_csv(PATH_RATINGS)
bookmarks = pd.read_csv(PATH_BOOKMARKS)

transactions = transactions[['element_uid', 'user_uid']]

items = items[['element_uid', 'duration', 'is_movie',
               'avail_p', 'avail_r', 'avail_s',
               'f1', 'f2', 'f3', 'f4',
               'f5_0', 'f5_1', 'f5_2', 'f5_3', 'f5_4', 'f5_5',
               'attr_cluster_0', 'attr_cluster_1', 'attr_cluster_2', 'attr_cluster_3',
               'attr_cluster_4', 'attr_cluster_5', 'attr_cluster_6', 'attr_cluster_7',
               'attr_cluster_8', 'attr_cluster_9', 'rating', 'bookmarks_freq',
               'watched_time', 'transactions_freq']]

users = users[['user_uid', 'ctr', 'transaction_made',
               'device_type_0_freq', 'device_type_1_freq', 'device_type_2_freq', 'device_type_3_freq',
               'device_manufacturer_0_freq', 'device_manufacturer_1_freq',
               'device_manufacturer_2_freq', 'device_manufacturer_3_freq',
               'consumption_mode_P_freq', 'consumption_mode_R_freq', 'consumption_mode_S_freq',
               'watched_time_sum', 'watched_time_mean', 'movie_watched_time_ratio', 'movie_ratio',
               'mean_items_duration', 'ratings_count', 'ratings_mean',
               'bookmarks_count']]

test_users = {k: 1 for k in test_users}
users['is_test'] = users['user_uid'].map(test_users)
users['is_test'] = users['is_test'].fillna(0)

#  Elements that users have already watched

already_watched = {k: list(v) for k, v in transactions.groupby('user_uid')['element_uid']}
rated = {k: list(v) for k, v in ratings.groupby('user_uid')['element_uid']}
saved = {k: list(v) for k, v in bookmarks.groupby('user_uid')['element_uid']}

for k, v in tqdm(already_watched.items()):
    new_val = v + rated.get(k, []) + saved.get(k, [])
    already_watched[k] = set(new_val)


#  Negative sampling

def negative_sample(user, size):
    banned = already_watched[user]
    choice = [x for x in items.element_uid.unique() if x not in banned]
    negatives = np.random.choice(choice, size=size)
    return negatives


len(users['user_uid'].unique())

#  Train data

transactions = transactions.merge(users[['user_uid', 'is_test']], how='left', on='user_uid')


def generate_dataset(size):
    df = transactions[transactions['is_test'] == 0].copy().sample(size)
    df = df[['element_uid', 'user_uid']]
    df['target'] = 1
    negative = []
    for _, row in tqdm(df.iterrows()):
        user_uid = row.user_uid
        element_uid = negative_sample(user_uid, 1)[0]
        negative.append({'element_uid': element_uid,
                         'user_uid': user_uid,
                         'target': 0})
    df = pd.concat([df, pd.DataFrame(negative)], axis=0)
    df = df.reset_index()
    return df


df = generate_dataset(500_000)