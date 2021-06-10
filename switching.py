import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, auc, precision_recall_curve, precision_score
from tqdm import tqdm
tqdm.pandas()

from drug_interactions.utils.calc_metrics import calc_average_precision_k, calc_mrr

def load_df(path, dataset_type=None):
    if dataset_type is not None:
        df = pd.read_csv(f'{path}/{dataset_type}.csv')

    else:
        df = pd.read_csv(path)

    df = df.sort_values(by=['Drug1_ID', 'Drug2_ID'], ascending=True).reset_index(drop=True)
    return df

def main():

    test = load_df('./data/csvs/data/test_all_similar.csv')

    newold_model = 'AFMP'
    newnew_model = 'CASTER'

    dataset_type = f'{newold_model}_OLD_{newnew_model}_NEW_SWITCHING'


    old_new_df = pd.read_csv(f'./data/csvs/results/All_Data/NewOldSimilar/{newold_model}.csv')
    new_new_df = pd.read_csv(f'./data/csvs/results/All_Data/NewNewSimilar/{newnew_model}.csv')

    test = old_new_df.append(new_new_df)

    df = test.copy()

    results = {}
    results['dataset_type'] = dataset_type
    label, prediction = df.label.tolist(), df.prediction.tolist()
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


    df = df.sort_values('prediction', ascending=False)
    df['class'] = df.prediction.progress_apply(lambda x: 1 if x > 0.5 else 0)

    top_k = list(range(1, 6)) + [10, 20, 50, 100, 200]
    for k in tqdm(top_k):
        results[f"Average Precision@{k}"] = calc_average_precision_k(df, k)

    print()

    results["MRR"] = calc_mrr(df)

    metrics = pd.read_csv(f'./data/csvs/results/All_Data/Tricks/Metrics.csv')
    metrics = metrics.append(pd.Series(results), ignore_index=True)
    metrics.to_csv(f'./data/csvs/results/All_Data/Tricks/Metrics.csv', index=False)

if __name__ == "__main__":
    main()