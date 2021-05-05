import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, precision_score
from tqdm import tqdm, trange
tqdm.pandas()

def load_preds(path):

    path = os.path.join(path, 'test_preds.csv')
    df = pd.read_csv(path)
    df['smiles'] = df['smiles'].progress_apply(eval)
    df['smile_a'] = df['smiles'].progress_apply(lambda smiles: smiles[0])
    df['smile_b'] = df['smiles'].progress_apply(lambda smiles: smiles[1])
    df = df.drop(columns=['smiles'])

    return df

def load_test():

    df = pd.read_csv('./data/csvs/smiles_test.csv')

    return df
def main():

    preds = load_preds('./models/shared/')
    test = load_test()
    test = test[(test.smile_a.isin(preds.smile_a)) & (test.smile_b.isin(preds.smile_b))]
    test = test.reset_index(drop=True)

    with open('./data/json/smiles_to_drug.json', 'r') as f:
        smiles_to_drug = json.load(f)

    preds['smile_a'] = preds['smile_a'].progress_apply(lambda s: smiles_to_drug[s])
    preds['smile_b'] = preds['smile_b'].progress_apply(lambda s: smiles_to_drug[s])

    
    print(preds.shape)
    print(preds.columns)
    print(preds.head())
    preds.to_csv('./CHEMPROP_preds.csv', index=False)

    # test['prediction'] = preds['label']


    # bce = log_loss(y_true=test.label.tolist(), y_pred=test.prediction.tolist())
    # auc = roc_auc_score(y_true=test.label.tolist(), y_score=test.prediction.tolist())
    # print(f'Test BCE: {round(bce, 4)}')
    # print(f'Test AUC: {round(auc, 4)}')

    # top_k = [1, 20, 40, 60, 80, 100]

    # test = test.sort_values('prediction', ascending=False)
    # test['class'] = test.prediction.progress_apply(lambda x: 1 if x > 0.5 else 0)

    # for k in top_k:
    #     y_true = np.array(test.label.tolist())
    #     y_pred = np.array(test['class'].tolist())
    #     print(f'Precision@{k}: {round(precision_score(y_true=y_true[:k], y_pred=y_pred[:k], zero_division=1), 4)}')

    # grouped_test = test.groupby('smile_a')

    # for k in trange(1, 6):
    #     average_precision = 0.0
    #     for (_, test_group) in tqdm(grouped_test, leave=False):
    #         try:
    #             y_true = np.array(test_group.label.tolist())
    #             y_pred = np.array(test_group['class'].tolist())
    #             average_precision += precision_score(y_true=y_true[:k], y_pred=y_pred[:k], zero_division=1)
    #         except:
    #             pass
    #     print(f'Average Precision@{k} {round(average_precision / test["smile_a"].nunique(), 4)}', end=' ')


if __name__ == "__main__":
    main()