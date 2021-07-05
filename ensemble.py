import os 

import pandas as pd
pd.set_option('display.max_columns', 10)
from tqdm import tqdm
tqdm.pandas()

from drug_interactions.utils.calc_metrics import calc_metrics

test_df = pd.read_csv('./data/csvs/data/test_all_chemprop_similar.csv')
test_df = test_df.sort_values(by=['Drug1_ID', 'Drug2_ID'], ascending=True).reset_index(drop=True)

path = './data/csvs/results/All_Data/AllEnsemble'

models = []
for model in os.listdir(path):
    if model != 'Metrics.csv':
        print(model)
        df_path = os.path.join(path, model)
        df = pd.read_csv(df_path)
        df = df.sort_values(by=['Drug1_ID', 'Drug2_ID'], ascending=True).reset_index(drop=True)
        model = model.split('.')[0]
        models.append(model)
        test_df[model] = df['prediction']

print(test_df.shape)
print(test_df.head())

print(models)
df = test_df.copy()
df['prediction'] = df[models].mean(axis=1)
results = {}
dataset_type = 'ENSEMBLE'

calc_metrics(dataset_type=dataset_type, path=path, df=df)