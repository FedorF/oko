import pandas as pd
import json
from matplotlib import pyplot as plt
import seaborn as sns
import missingno as msno
import numpy as np
from scipy import sparse


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

#  Catalogue EDA

msno.matrix(catalogue, inline=True, sparkline=True, figsize=(20, 10), sort=None)  # Null значения отсутствуют

#  Catalogue types of content
catalogue.type.value_counts()

sns.countplot(catalogue['type'])
plt.savefig(PATH_PLOTS + '/item_types.png')

#  Catalogue availability
catalogue[catalogue['availability'].map(lambda x: len(x)) == 0].shape  # 2097 элементов
#  у которых нет ни одного значения доступности
#  из статьи на хабре: возможно у кинотаетра за
#  кончился контракт на фильм с правообладателем

catalogue['availability_purchase'] = catalogue['availability'].map(lambda x: 1 if 'purchase' in x else 0)
catalogue['availability_rent'] = catalogue['availability'].map(lambda x: 1 if 'rent' in x else 0)
catalogue['availability_subscription'] = catalogue['availability'].map(lambda x: 1 if 'subscription' in x else 0)

catalogue[catalogue['availability_purchase'] == 1].shape[0]  # 7824
catalogue[catalogue['availability_rent'] == 1].shape[0]  # 5006
catalogue[catalogue['availability_subscription'] == 1].shape[0]  # 6781

#  Catalogue duration
catalogue['duration'].describe()
plt.hist(catalogue[catalogue['duration'] < 5]['duration'])
plt.show()
sns.distplot(catalogue[catalogue['duration'] < 50]['duration'])
plt.savefig(PATH_PLOTS + '/item_duration.png')

catalogue['duration_is_zero'] = catalogue['duration'].map(lambda x: 1 if x == 0 else 0)  # 114 items

#  Catalogue anonymous features
catalogue[['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']].describe().T
sns.heatmap(catalogue[['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']].corr())
plt.show()

len(catalogue['feature_3'].value_counts())

features = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']


def get_plots(feature):
    plt.ion()
    plt.figure()
    plt.subplot(2, 1, 1)
    sns.distplot(catalogue[feature])
    plt.subplot(2, 1, 2)
    sns.distplot(np.log(catalogue[feature]))
    plt.ioff()
    plt.savefig(PATH_PLOTS + '/' + feature + '.png')


get_plots(features[4])

plt.ion()
plt.figure()
sns.distplot(catalogue['feature_5'])
plt.ioff()
# plt.title('categorical')
plt.savefig(PATH_PLOTS + '/' + 'feature_5' + '.png')

catalogue['feature_3'].value_counts()

feature_5 = catalogue['feature_5'].unique()
print('exp(feature_5) = {}'.format(sorted(np.exp(feature_5 + 1))))

print('{}'.format(sorted(np.log(feature_5 + 2))))

#  Catalogue attributes features

element_attributes = set(x for y in catalogue.attributes.values for x in y)
print('attributes count: {}'.format(len(element_attributes)))

#  Ratings
plt.ion()
plt.figure()
sns.distplot(ratings['rating'])
plt.ioff()
plt.savefig(PATH_PLOTS + '/' + 'ratings_hist' + '.png')

#  Bookmarks
bookmarks[bookmarks['user_uid'][test_users]]

#  Test users
test_users

#  Transactions


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

sparsity = consumption_matrix.nnz / (consumption_matrix.shape[0] * consumption_matrix.shape[1])
print('Sparsity: %.6f' % sparsity)
