from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import json
from scipy import sparse
tqdm.pandas(desc="my bar!")

def read_json(path):
    with open(path, 'r') as f:
        output = json.load(f)
    return output

def write_json(path, x):
    with open(path, 'w') as f:
        f.write(json.dumps(x))


PATH_TEST = './data/test_users.json'
PATH_TRANSACTIONS = './data/transactions.csv'
PATH_RATINGS = './data/ratings.csv'
PATH_BOOKMARKS = './data/bookmarks.csv'
PATH_CATALOGUE = './data/catalogue.json'

transactions = pd.read_csv(PATH_TRANSACTIONS)
ratings = pd.read_csv(PATH_RATINGS)
bookmarks = pd.read_csv(PATH_BOOKMARKS)
catalogue = pd.read_csv('./data/catalogue.csv')
test_users = read_json(PATH_TEST)['users']

#  Найдем транзакции, которые подходят под условия задачи для потребленных товаров
transactions['consumption_mode'].unique()
transactions['is_consumed'] = 0
transactions['is_consumed'] = transactions['consumption_mode'].map({'P': 1,
                                                                    'R': 1,
                                                                    'S': 0})

transactions = transactions.merge(catalogue[['element_uid', 'type', 'duration']], on='element_uid')

mask = (transactions['consumption_mode'] == 'S') & \
                    ((transactions['type'] == 'movie') | (transactions['type'] == 'multipart_movie')) & \
                    (transactions['watched_time'] > (transactions['duration'] / 2))

transactions.loc[mask, ['is_consumed']] = 1

mask_series_subscribe = (transactions['consumption_mode'] == 'S') & \
                        (transactions['type'] == 'series') & \
                        (transactions['watched_time'] > (transactions['duration'] / 3))

transactions.loc[mask_series_subscribe, ['is_consumed']] = 1

#  Построим матрицу user_item

transactions['user_uid'] = transactions['user_uid'].astype('category')
transactions['element_uid'] = transactions['element_uid'].astype('category')

consumption_matrix = sparse.coo_matrix(
    (transactions['is_consumed'], (
            transactions['element_uid'].cat.codes.copy(),
            transactions['user_uid'].cat.codes.copy()
        )
    )
)

consumption_matrix = consumption_matrix.tocsr()
sparse.save_npz('./source/user_item_matrix.npz', consumption_matrix)

#  Mapping from original id to inner
user_uid_to_cat = dict(zip(
    transactions['user_uid'].cat.categories,
    range(len(transactions['user_uid'].cat.categories))
))

element_uid_to_cat = dict(zip(
    transactions['element_uid'].cat.categories,
    range(len(transactions['element_uid'].cat.categories))
))

write_json('./mapping/u2u', user_uid_to_cat)
write_json('./mapping/i2i', element_uid_to_cat)


#  Save elements that user wouldn't convert for sure
is_test = pd.DataFrame(test_users)
is_test['is_test'] = 1
is_test.columns = ['user_uid', 'is_test']

transactions['is_test'] = 0
transactions = transactions.merge(is_test, on='user_uid', how='left')
transactions['is_test'] = transactions['is_test_y'].fillna(0)
transactions = transactions[['element_uid', 'user_uid', 'consumption_mode', 'ts', 'watched_time',
       'device_type', 'device_manufacturer', 'is_consumed', 'type', 'duration', 'is_test']]

filtered_elements = {k:v for k, v in transactions[transactions['is_test'] == 1].groupby('user_uid')['element_uid']}
filtered_elements.items()
filtered_elements_cat = {k: [element_uid_to_cat.get(x, None) for x in v] for k, v in filtered_elements.items()}
