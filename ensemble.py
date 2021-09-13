import os 

import pandas as pd
pd.set_option('display.max_columns', 10)
from tqdm import tqdm
tqdm.pandas()

from drug_interactions.utils.calc_metrics import calc_metrics

test_df = pd.read_csv('./data/csvs/data/test_new_old_chemprop_similar.csv')
test_df = test_df.sort_values(by=['Drug1_ID', 'Drug2_ID'], ascending=True).reset_index(drop=True)

path = './data/csvs/results/All_Data/NewOldSimilarityEnsemble'

models = {
    'LAFMP': 1,
    'CHEMPROP': 1/2,
    'CASTER': 1/3,
    'AFMP': 1/4,
    'ONEHOT_SMILES': 1/5,
    'CHAR_2_VEC': 1/6,

}
for model, weight in models.items():
    if model != 'Metrics.csv':
        print(model)
        df_path = os.path.join(path, f'{model}.csv')
        df = pd.read_csv(df_path)
        df = df.sort_values(by=['Drug1_ID', 'Drug2_ID'], ascending=True).reset_index(drop=True)
        # model = model.split('.')[0]
        df['prediction'] *= weight
        test_df[model] = df['prediction']

print(test_df.shape)
print(test_df.head())

print(models)
df = test_df.copy()
df['prediction'] = df[models].sum(axis=1)
df['prediction'] /= sum(models.values())
results = {}
dataset_type = 'RANKED_ENSEMBLE'

calc_metrics(dataset_type=dataset_type, path=path, df=df)