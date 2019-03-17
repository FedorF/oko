from implicit.als import AlternatingLeastSquares
import json
from tqdm import tqdm
from scipy import sparse
import pandas as pd

def read_json(path):
    with open(path, 'r') as f:
        output = json.load(f)
    return output

PATH_TEST = './data/test_users.json'
PATH_BOOKMARKS = './data/bookmarks.csv'


test_users = read_json(PATH_TEST)['users']
bookmarks = pd.read_csv(PATH_BOOKMARKS)

user_uid_to_cat = read_json('./mapping/u2u')
element_uid_to_cat = read_json('./mapping/i2i')
sparse_matrix = sparse.load_npz('./source/user_item_matrix.npz')

model = AlternatingLeastSquares(factors=10)
model.fit(sparse_matrix)

# recommend items for a user
user_items = sparse_matrix.T.tocsr()

result = {}
for user_uid in tqdm(test_users):
    # transform user_uid to model's internal user category
    try:
        user_cat = user_uid_to_cat[str(user_uid)]
    except LookupError:
        continue

    # perform inference
    recs = model.recommend(
        user_cat,
        user_items,
        N=20
    )

    # drop scores and transform model's internal element category to element_uid for every prediction
    # also convert np.uint64 to int so it could be json serialized later
    result[user_uid] = [int(transactions['element_uid'].cat.categories[i]) for i, _ in recs]
print(len(result))


#  Если модель не учла некоторых юзеров. То рекомендуем им то, что в закладках
us2book = bookmarks.groupby('user_uid')['element_uid'].apply(lambda x: x.tolist()).to_dict()
most_popular = transactions.groupby('element_uid')['element_uid'].count()[:20].items(), key=lambda x: x[0]

for user_id in test_users:
    if user_id in result.keys():
        continue
    else:
        result[user_id] = us2book[user_id][:20]

with open('./submisions/answer.json', 'w') as f:
    json.dump(result, f)
