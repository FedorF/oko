import pandas as pd
import json
from collections import Counter
from itertools import combinations
import numpy as np


def read_json(path):
    with open(path, 'r') as f:
        output = json.load(f)
    return output


PATH_TEST = './data/test_users.json'
PATH_TRANSACTIONS = './data/transactions.csv'
PATH_RATINGS = './data/ratings.csv'
PATH_BOOKMARKS = './data/bookmarks.csv'
PATH_CATALOGUE = './data/catalogue.json'
PATH_PLOTS = './plots'

transactions = pd.read_csv(PATH_TRANSACTIONS)
ratings = pd.read_csv(PATH_RATINGS)
bookmarks = pd.read_csv(PATH_BOOKMARKS)
catalogue = pd.DataFrame.from_dict(read_json(PATH_CATALOGUE), orient='index')
# catalogue = pd.read_csv('./data/catalogue.csv')
test_users = read_json(PATH_TEST)['users']

coocurancy = pd.concat([transactions[['user_uid', 'element_uid']], bookmarks[['user_uid', 'element_uid']],
                        ratings[ratings['rating'] > 5][['user_uid', 'element_uid']]])
coocurancy = [x for x in coocurancy.groupby('user_uid')['element_uid'].apply(lambda x: x.tolist()).to_dict().values()]

coocurancy = [set(x) for x in coocurancy]

cnt = Counter()
for x in coocurancy:
    for el in x:
        cnt[el] += 1

cnt_xy = Counter()
for b in coocurancy:
    for xy in combinations(b, 2):
        cnt_xy[xy] += 1


for k, v in cnt.items():
    cnt[k] = np.log(v / len(cnt))

for k, v in cnt_xy.items():
    cnt_xy[k] = np.log(v / len(cnt_xy))

