import pandas as pd
import json
from matplotlib import pyplot as plt
import seaborn as sns
import missingno as msno
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import MiniBatchKMeans


def read_json(path):
    with open(path, 'r') as f:
        output = json.load(f)
    return output


PATH_TEST = './data/test_users.json'
PATH_TRANSACTIONS = './data/transactions.csv'
PATH_RATINGS = './data/ratings.csv'
PATH_BOOKMARKS = './data/bookmarks.csv'
PATH_CATALOGUE = './data/catalogue.json'
PATH_ELEMENTS = './features/items_features.csv'

transactions = pd.read_csv(PATH_TRANSACTIONS)
ratings = pd.read_csv(PATH_RATINGS)
bookmarks = pd.read_csv(PATH_BOOKMARKS)
catalogue = pd.read_csv('./data/catalogue.csv')
items = pd.read_csv(PATH_ELEMENTS)

#  Create users DataFrame
users = transactions.copy()
users = users.drop_duplicates('user_uid')
users = users.reset_index(drop=True)

#  Transactions
mean_watched_time = transactions.groupby('user_uid')['watched_time'].mean().to_dict()
users['mean_watched_time'] = users.user_uid.map(mean_watched_time)

sum_watched_time = transactions.groupby('user_uid')['watched_time'].sum().to_dict()
users['sum_watched_time'] = users.user_uid.map(sum_watched_time)

user_ctr = -np.log(transactions.groupby('user_uid')['user_uid'].count() / transactions.shape[0])
users['ctr'] = users['user_uid'].map(user_ctr)

# Device type freq

dev_type = {0: 0,
            3: 1,
            5: 2,
            1: 3,
            4: 3,
            6: 3,
            2: 3}

transactions['device_type'] = transactions.device_type.map(dev_type)
transactions = pd.concat([transactions, pd.get_dummies(transactions['device_type'], prefix='device_type')], axis=1)

users['transaction_made'] = users['user_uid'].map(transactions.groupby('user_uid')['user_uid'].count().to_dict())

users['device_type_0_count'] = users['user_uid'].map(transactions.groupby('user_uid')['device_type_0'].sum().to_dict())
users['device_type_1_count'] = users['user_uid'].map(transactions.groupby('user_uid')['device_type_1'].sum().to_dict())
users['device_type_2_count'] = users['user_uid'].map(transactions.groupby('user_uid')['device_type_2'].sum().to_dict())
users['device_type_3_count'] = users['user_uid'].map(transactions.groupby('user_uid')['device_type_3'].sum().to_dict())

users['device_type_0_freq'] = users['device_type_0_count'] / users['transaction_made']
users['device_type_1_freq'] = users['device_type_1_count'] / users['transaction_made']
users['device_type_2_freq'] = users['device_type_2_count'] / users['transaction_made']
users['device_type_3_freq'] = users['device_type_3_count'] / users['transaction_made']

# Device manufacturer freq


man_type = {k: 3 for k in transactions.device_manufacturer.value_counts().to_dict().keys()}
man_type[50] = 0
man_type[11] = 1
man_type[99] = 2
transactions['device_manufacturer'] = transactions['device_manufacturer'].map(man_type)
transactions = pd.concat(
    [transactions, pd.get_dummies(transactions['device_manufacturer'], prefix='device_manufacturer')], axis=1)

users['device_manufacturer_0_count'] = users['user_uid'].map(
    transactions.groupby('user_uid')['device_manufacturer_0'].sum().to_dict())
users['device_manufacturer_1_count'] = users['user_uid'].map(
    transactions.groupby('user_uid')['device_manufacturer_1'].sum().to_dict())
users['device_manufacturer_2_count'] = users['user_uid'].map(
    transactions.groupby('user_uid')['device_manufacturer_2'].sum().to_dict())
users['device_manufacturer_3_count'] = users['user_uid'].map(
    transactions.groupby('user_uid')['device_manufacturer_3'].sum().to_dict())

users['device_manufacturer_0_freq'] = users['device_manufacturer_0_count'] / users['transaction_made']
users['device_manufacturer_1_freq'] = users['device_manufacturer_1_count'] / users['transaction_made']
users['device_manufacturer_2_freq'] = users['device_manufacturer_2_count'] / users['transaction_made']
users['device_manufacturer_3_freq'] = users['device_manufacturer_3_count'] / users['transaction_made']

#  Consumption type freq

transactions.consumption_mode.value_counts()
transactions = pd.concat([transactions, pd.get_dummies(transactions['consumption_mode'], prefix='consumption_mode')],
                         axis=1)

users['consumption_mode_P_count'] = users['user_uid'].map(
    transactions.groupby('user_uid')['consumption_mode_P'].sum().to_dict())
users['consumption_mode_R_count'] = users['user_uid'].map(
    transactions.groupby('user_uid')['consumption_mode_R'].sum().to_dict())
users['consumption_mode_S_count'] = users['user_uid'].map(
    transactions.groupby('user_uid')['consumption_mode_S'].sum().to_dict())

users['consumption_mode_P_freq'] = users['consumption_mode_P_count'] / users['transaction_made']
users['consumption_mode_R_freq'] = users['consumption_mode_R_count'] / users['transaction_made']
users['consumption_mode_S_freq'] = users['consumption_mode_S_count'] / users['transaction_made']



#  Items types freq and watched time rate
transactions = transactions.merge(items[['element_uid', 'is_movie']], how='left', on='element_uid')
users['movie_watched_time_sum'] = users['user_uid'].map(transactions[transactions['is_movie'] == 1].groupby('user_uid')['watched_time'].sum().to_dict())
users['movie_watched_time_ratio'] = users['movie_watched_time_sum'] / users['watched_time_sum']


users['movie_count'] = users['user_uid'].map(transactions[transactions['is_movie'] == 1].groupby('user_uid')['element_uid'].count().to_dict())
users['movie_ratio'] = users['movie_count'] / users['transaction_made']



#  Items duration user's preferences. Count mean duration of elements watched by users
transactions = transactions.merge(items[['element_uid', 'duration']], how='left', on='element_uid')
users['mean_items_duration'] = users['user_uid'].map(transactions.groupby('user_uid')['duration'].mean().to_dict())


#  Ratings
users['ratings_count'] = users['user_uid'].map(ratings.groupby('user_uid')['user_uid'].count().to_dict())
users['ratings_count'] = users['ratings_count'].fillna(0)

users['ratings_mean'] = users['user_uid'].map(ratings.groupby('user_uid')['rating'].mean().to_dict())
mean_ratings = ratings.groupby('user_uid')['rating'].mean().mean()
users['ratings_mean'] = users['ratings_mean'].fillna(mean_ratings)


#  Bookmarks
users['bookmarks_count'] = users['user_uid'].map(bookmarks.groupby('user_uid')['user_uid'].count().to_dict())
users['bookmarks_count'] = users['bookmarks_count'].fillna(0)

users = users[['user_uid', 'ctr', 'transaction_made', 'device_type_0_freq',
       'device_type_1_freq', 'device_type_2_freq', 'device_type_3_freq',
       'device_manufacturer_0_freq', 'device_manufacturer_1_freq',
       'device_manufacturer_2_freq', 'device_manufacturer_3_freq',
       'consumption_mode_P_freq', 'consumption_mode_R_freq',
       'consumption_mode_S_freq', 'movie_watched_time_sum',
       'watched_time_sum', 'watched_time_mean',
       'movie_watched_time_ratio', 'movie_ratio',
       'mean_items_duration', 'ratings_count', 'ratings_mean',
       'bookmarks_count']]

users.to_csv('./features/users_features.csv', index=False)