import os 

import pandas as pd
pd.set_option('display.max_columns', 10)
from tqdm import tqdm
tqdm.pandas()

from drug_interactions.utils.calc_metrics import calc_metrics

test_df = pd.read_csv('./data/csvs/data/test_all_chemprop_similar.csv')
test_df = test_df.sort_values(by=['Drug1_ID', 'Drug2_ID'], ascending=True).reset_index(drop=True)

path = './data/csvs/results/All_Data/All'
models = ['LAFMP', 'CHAR_2_VEC', 'ONEHOT_SMILES', 'CASTER', 'CHEMPROP', 'AFMP']
similarities = ['EditDistance', 'NLCS', 'TF', 'ChempropSimilar', 'Similar']

for model in (t := tqdm(models)):
    t.set_description(model)
    for similarity in (ts := tqdm(similarities, leave=False)):
        ts.set_description(similarity)
        similarity_path = path + similarity
        if model == 'ENSEMBLE':
            model_a_path = os.path.join(similarity_path, f'LAFMP.csv')
            model_b_path = os.path.join(similarity_path, f'CHEMPROP.csv')
            df_a = pd.read_csv(model_a_path)
            df_b = pd.read_csv(model_b_path)
            df_a = df_a.sort_values(by=['Drug1_ID', 'Drug2_ID'], ascending=True).reset_index(drop=True)
            df_b = df_b.sort_values(by=['Drug1_ID', 'Drug2_ID'], ascending=True).reset_index(drop=True)
            # model = model.split('.')[0]
            ensemble_prediction = (df_a['prediction'] + df_b['prediction']) / 2
            test_df[similarity] = ensemble_prediction

        else:
            df_path = os.path.join(similarity_path, f'{model}.csv')
            df = pd.read_csv(df_path)
            df = df.sort_values(by=['Drug1_ID', 'Drug2_ID'], ascending=True).reset_index(drop=True)
            # model = model.split('.')[0]
            test_df[similarity] = df['prediction']

    df = test_df.copy()
    df['prediction'] = df[similarities].mean(axis=1)
    results = {}
    dataset_type = f'{model}'

    save_path = path + "SimilarityEnsemble/"
    df = df[['Drug1_ID', 'Drug2_ID', 'prediction']]
    df.to_csv(save_path + f'{model}.csv', index=False)
    # calc_metrics(dataset_type=dataset_type, path=save_path, df=df)