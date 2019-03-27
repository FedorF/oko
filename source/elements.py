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
PATH_PLOTS = './plots'

transactions = pd.read_csv(PATH_TRANSACTIONS)
ratings = pd.read_csv(PATH_RATINGS)
bookmarks = pd.read_csv(PATH_BOOKMARKS)
catalogue = pd.read_csv('./data/catalogue.csv')

#  Catalogue types of content
catalogue.type.value_counts()
is_movie = {'movie': 1,
            'multipart_movie': 1,
            'series': 0}

catalogue['is_movie'] = catalogue['type'].map(is_movie)


#  Catalogue availability
catalogue['avail_p'] = catalogue['availability'].map(lambda x: 1 if 'purchase' in x else 0)
catalogue['avail_r'] = catalogue['availability'].map(lambda x: 1 if 'rent' in x else 0)
catalogue['avail_s'] = catalogue['availability'].map(lambda x: 1 if 'subscription' in x else 0)


#  Catalogue anonymous features

catalogue['feature_1'] = catalogue['feature_1'].map(np.log10)
rscaler = RobustScaler()
stdscaler = StandardScaler()
maxminscaler = MinMaxScaler()

f1_norm = maxminscaler.fit_transform(catalogue['feature_1'].values.reshape(-1, 1))
catalogue['f1'] = f1_norm


f2_norm = maxminscaler.fit_transform(catalogue['feature_2'].values.reshape(-1, 1))
catalogue['f2'] = f2_norm


catalogue['f3'] = catalogue['feature_3'].map(lambda x: 48 if x == 50 else x)


f4_norm = maxminscaler.fit_transform(catalogue['feature_4'].values.reshape(-1, 1))
catalogue['f4'] = f4_norm


f5_mapping = {k: v for v, k in enumerate(sorted((catalogue['feature_5'] + 1).unique()))}
catalogue['feature_5'] = catalogue['feature_5'] + 1
catalogue['f5'] = catalogue['feature_5'].map(f5_mapping)
catalogue = pd.concat([catalogue, pd.get_dummies(catalogue['f5'], prefix='f5')], axis=1)


#  Catalogue attributes features

catalogue.attributes = catalogue.attributes.map(lambda x: x[1:-1].replace(' ', ''))

vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
attr_vect = vectorizer.fit_transform(catalogue.attributes.values)

inertia = []
for k in range(1, 50):
    print(k)
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=19).fit(attr_vect)
    inertia.append(np.sqrt(kmeans.inertia_))
plt.ion()
plt.figure()
plt.plot(range(1, 50), inertia, marker='s')
plt.grid()
plt.ioff()
kmeans = MiniBatchKMeans(n_clusters=10, random_state=19).fit(attr_vect)

catalogue['attr_cluster'] = kmeans.labels_
catalogue = pd.concat([catalogue, pd.get_dummies(catalogue['attr_cluster'], prefix='attr_cluster')], axis=1)



#  Ratings
catalogue = catalogue.merge(ratings.groupby('element_uid')['rating'].mean(), how='left', on='element_uid')
catalogue['rating'] = catalogue['rating'].fillna(catalogue.rating.mean())



#  Bookmarks
bookmarks_count = bookmarks.shape[0]
bookmarks_freq = -np.log(bookmarks.groupby('element_uid')['element_uid'].count() / bookmarks_count)
catalogue['bookmarks_freq'] = catalogue['element_uid'].map(bookmarks_freq.to_dict())
catalogue['bookmarks_freq'] = catalogue['bookmarks_freq'].fillna(-np.log(1 / bookmarks_count))



#  Transactions
watched_time = transactions.groupby('element_uid')['watched_time'].mean().to_dict()
catalogue['watched_time'] = catalogue.element_uid.map(watched_time)
catalogue['watched_time'] = catalogue['watched_time'].fillna(0)


transactions_count = transactions.shape[0]
transactions_freq = -np.log(transactions.groupby('element_uid')['element_uid'].count() / transactions_count)
catalogue['transactions_freq'] = catalogue['element_uid'].map(transactions_freq.to_dict())
catalogue['transactions_freq'] = catalogue['transactions_freq'].fillna(-np.log(1 / transactions_count))
