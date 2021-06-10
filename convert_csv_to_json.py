import json
import pandas as pd
from tqdm import tqdm

test = pd.read_csv('./data/csvs/data/test_new_old.csv')
new_drugs = test.Drug1_ID.tolist()
old_drugs = test.Drug2_ID.tolist()

similar_matrix = pd.read_csv('./data/csvs/similar_matrix/simi_matrix.csv', index_col=0)

similar_drugs_columns = similar_matrix.columns.tolist()
similar_drugs_columns = [d for d in similar_drugs_columns if d in new_drugs]

map_ = {}

for drug in tqdm(similar_drugs_columns):
    col = similar_matrix[drug]
    col = col.sort_values(ascending=False)

    col = col[(col.index != drug) & (col.index.isin(old_drugs))]
    map_[drug] = [list(pair) for pair in list(zip(col.index, col))]

with open('./foo.txt', 'w') as f:
    json.dump(map_, f)