import json
import os

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 10)
from sklearn.metrics import log_loss, roc_auc_score, precision_score, confusion_matrix
from tqdm import tqdm, trange
tqdm.pandas()

test_df = pd.read_csv('./data/csvs/smiles_test_drug.csv')
test_df = test_df.sort_values(by=['smile_a', 'smile_b'], ascending=True).reset_index(drop=True)

onehot = pd.read_csv('./data/csvs/results/ONEHOT_SMILES.csv')
onehot = onehot.sort_values(by=['smile_a', 'smile_b'], ascending=True).reset_index(drop=True)

chemprop = pd.read_csv('./data/csvs/results/CHEMPROP.csv')
chemprop = chemprop.sort_values(by=['smile_a', 'smile_b'], ascending=True).reset_index(drop=True)

char2vec = pd.read_csv('./data/csvs/results/CHAR_2_VEC.csv')
char2vec = char2vec.sort_values(by=['smile_a', 'smile_b'], ascending=True).reset_index(drop=True)

deepsmiles = pd.read_csv('./data/csvs/results/DEEP_SMILES.csv')
deepsmiles = deepsmiles.sort_values(by=['smile_a', 'smile_b'], ascending=True).reset_index(drop=True)

coldstart = pd.read_csv('./data/csvs/results/COLD_START.csv')
coldstart = deepsmiles.sort_values(by=['smile_a', 'smile_b'], ascending=True).reset_index(drop=True)

test_df = test_df[(test_df.smile_a.isin(chemprop.smile_a)) & (test_df.smile_b.isin(chemprop.smile_b))]
onehot = onehot[(onehot.smile_a.isin(chemprop.smile_a)) & (onehot.smile_b.isin(chemprop.smile_b))]
char2vec = char2vec[(char2vec.smile_a.isin(chemprop.smile_a)) & (char2vec.smile_b.isin(chemprop.smile_b))]
deepsmiles = deepsmiles[(deepsmiles.smile_a.isin(chemprop.smile_a)) & (deepsmiles.smile_b.isin(chemprop.smile_b))]
coldstart = coldstart[(coldstart.smile_a.isin(chemprop.smile_a)) & (coldstart.smile_b.isin(chemprop.smile_b))]

print(test_df.shape)
print(onehot.shape)
print(chemprop.shape)
print(char2vec.shape)
print(deepsmiles.shape)
print(coldstart.shape)

test_df['ONEHOT_SMILES'] = onehot['prediction']
test_df['CHAR_2_VEC'] = char2vec['prediction']
test_df['COLD_START'] = coldstart['prediction']
test_df['CHEMPROP'] = chemprop['prediction']
test_df['DEEP_SMILES'] = deepsmiles['prediction']
test_df['models_mean'] = test_df[['CHAR_2_VEC', 'ONEHOT_SMILES', 'COLD_START', 'DEEP_SMILES']].mean(axis=1)

test_df = test_df.dropna()
print(test_df.shape)

results_df = pd.DataFrame(columns=['Threshold',
                                    'auc',
                                    'bce',
                                    'Precision at 1',
                                    'Precision at 20',
                                    'Precision at 40',
                                    'Precision at 60',
                                    'Precision at 80',
                                    'Precision at 100',
                                    'Average precision at 1', 
                                    'Average precision at 2',
                                    'Average precision at 3',
                                    'Average precision at 4',
                                    'Average precision at 5',
                                    ])

for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:

    results = {}

    results["Threshold"] = threshold

    print('-----------------')
    print(f'{threshold=}')
    print('-----------------')
    
    # test_df['above'] = test_df['models_mean'].apply(lambda x: 1 if x > threshold or x < (1 - threshold) else 0)
    # test_df = test_df[test_df['above'] == 1]
    chemprop_preds = test_df['CHEMPROP'].tolist()
    models_mean = test_df['models_mean'].tolist()

    # models_mean = [mean if mean > threshold else threshold for mean in models_mean]
    preds = [mean if mean < threshold else chem_preds for mean, chem_preds in zip(models_mean, chemprop_preds)]

    test_df['thresh_preds'] = preds
    # print(test_df.shape)
    test_df = test_df.dropna()
    # print(test_df.shape)
    results["bce"] = round(log_loss(y_true=test_df.label.tolist(), y_pred=test_df.thresh_preds.tolist()), 4)
    results["auc"]  = round(roc_auc_score(y_true=test_df.label.tolist(), y_score=test_df.thresh_preds.tolist()), 4)
    print(f'Test BCE: {results["bce"]}')
    print(f'Test AUC: {results["auc"]}')

    top_k = [1, 20, 40, 60, 80, 100]

    test_df = test_df.sort_values('thresh_preds', ascending=False)
    test_df['class'] = test_df.thresh_preds.apply(lambda x: 1 if x > 0.5 else 0)

    for k in top_k:
        y_true = np.array(test_df.label.tolist())
        y_pred = np.array(test_df['class'].tolist())
        precision = round(precision_score(y_true=y_true[:k], y_pred=y_pred[:k], zero_division=1), 4)
        results[f"Precision at {k}"] = precision
        print(f'Precision@{k}: {precision}')

    grouped_test = test_df.groupby('smile_a')

    for k in range(1, 6):
        average_precision = 0.0
        for (_, test_group) in grouped_test:
            try:
                y_true = np.array(test_group.label.tolist())
                y_pred = np.array(test_group['class'].tolist())
                average_precision += precision_score(y_true=y_true[:k], y_pred=y_pred[:k], zero_division=1)
            except:
                pass
        average_precision = round(average_precision / test_df["smile_a"].nunique(), 4)
        results[f"Average precision at {k}"] = average_precision
        print(f'Average Precision@{k} {average_precision}')

    series = pd.Series(results)
    results_df = results_df.append(series, ignore_index=True)

results_df.to_csv('./results_all.csv', index=False)