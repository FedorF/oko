import pandas as pd
import json


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
PATH_SUBMISSION = './submissions'

transactions = pd.read_csv(PATH_TRANSACTIONS)
ratings = pd.read_csv(PATH_RATINGS)
bookmarks = pd.read_csv(PATH_BOOKMARKS)
catalogue = pd.DataFrame.from_dict(read_json(PATH_CATALOGUE), orient='index')
test_users = read_json(PATH_TEST)['users']

#  Bookmarks


#  Test users
bookmarks['is_test'] = bookmarks['user_uid'].map(lambda x: 1 if x in test_users else 0)
user2bookmark = {k: [] for k in test_users}
for _, row in bookmarks[bookmarks['is_test'] == 1].iterrows():
    user = row['user_uid']
    item = row['element_uid']
    user2bookmark[user].append(item)

max(len(x) for x in user2bookmark.values())

len([x for x in user2bookmark.values() if len(x) < 1])

#  Transactions
element_conv_cnt = transactions.groupby('element_uid')['user_uid'].count().to_dict()
most_popular_elements = [x[0] for x in sorted(element_conv_cnt.items(), key=lambda x: x[1], reverse=True)[:20]]


#  Example

with open('./data/answer.json', 'r') as f:
    answer = json.load(f)

#  Submission

for u, it in user2bookmark.items():
    if u in answer.keys():
        user2bookmark[u] = answer[u]
        continue
    if len(it) < 1:
        user2bookmark[u] = most_popular_elements
    elif len(it) > 20:
        user2bookmark[u] = it[:20]



def make_submission(path, pred):
    with open(path, 'w') as f:
        f.write(json.dumps(pred))

make_submission(PATH_SUBMISSION + '/bookmarks_ratings_knn.json', user2bookmark)

