import os 

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 10)
from sklearn.metrics import log_loss, roc_auc_score, precision_score, precision_recall_curve, auc
from tqdm import tqdm, trange
tqdm.pandas()

test_df = pd.read_csv('./data/csvs/data/test_all_similar.csv')
test_df = test_df.sort_values(by=['Drug1_ID', 'Drug2_ID'], ascending=True).reset_index(drop=True)

path = './data/csvs/results/All_Data/AllSimilar'

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
df['mean'] = df[models].mean(axis=1)
results = {}
results['dataset_type'] = 'ENSEMBLE'

print(df.isna().sum())
label, prediction = df.label.tolist(), df['mean'].tolist()
bce = round(log_loss(y_true=label, y_pred=prediction), 4)
roc_auc = round(roc_auc_score(y_true=label, y_score=prediction), 4)
precision, recall, _ = precision_recall_curve(y_true=label, probas_pred=prediction)
pr_auc = round(auc(recall, precision), 4)

results['Log Loss'] = bce
results['AUC'] = roc_auc
results['PR AUC'] = pr_auc

print(f'Test BCE: {bce}')
print(f'Test AUC: {roc_auc}')
print(f'Test PR-AUC: {pr_auc}')

top_k = list(range(1, 6)) + [10, 20, 50, 100, 200]

df = df.sort_values('mean', ascending=False)
df['class'] = df['mean'].progress_apply(lambda x: 1 if x > 0.5 else 0)

grouped_test = df.groupby('Drug1_ID')

for k in tqdm(top_k):
    average_precision = 0.0
    for (_, smile_group) in tqdm(grouped_test, leave=False):
        smile_group = smile_group.sort_values('mean', ascending=False)
        try:
            y_true = np.array(smile_group.label.tolist())
            y_pred = np.array(smile_group['class'].tolist())
            average_precision += precision_score(y_true=y_true[:k], y_pred=y_pred[:k], zero_division=0)
        except Exception:
            pass
    average_precision = round(average_precision / df["Drug1_ID"].nunique(), 4)
    print(f'Average Precision@{k} {average_precision}', end=' ')
    results[f"Average Precision@{k}"] = average_precision

print()

mrr = []
for (_, dg) in tqdm(grouped_test, leave=False):
    dg['binary'] = dg['mean'].apply(lambda x: 1 if x >= 0.5 else 0)
    dg = dg.sort_values('mean', ascending=False)
    dg.reset_index(drop=True, inplace=True)
    first_hit = dg[(dg['label'] == 1) & (dg['binary'] == 1)]
    if first_hit.shape[0] == 0:
        mrr.append(0)
    else:
        mrr.append(1 / (first_hit.index[0] + 1))

mrr = round(np.array(mrr).mean(), 4)
print(f'Test MRR: {mrr}')
results["MRR"] = mrr

metrics = pd.read_csv(f'{path}/Metrics.csv')
metrics = metrics.append(pd.Series(results), ignore_index=True)
metrics.to_csv(f'{path}/Metrics.csv', index=False)
