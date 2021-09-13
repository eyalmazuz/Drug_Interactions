import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, auc, precision_score, precision_recall_curve, log_loss
from tqdm import tqdm
tqdm.pandas()

def calc_mrr(df):
    grouped_test = df.groupby('Drug1_ID')
    mrr = []
    for (_, dg) in tqdm(grouped_test, leave=False):
        dg['binary'] = dg['prediction'].apply(lambda x: 1 if x >= 0.5 else 0)
        dg = dg.sort_values('prediction', ascending=False)
        dg.reset_index(drop=True, inplace=True)
        first_hit = dg[(dg['label'] == 1) & (dg['binary'] == 1)]
        if first_hit.shape[0] == 0:
            mrr.append(0)
        else:
            mrr.append(1 / (first_hit.index[0] + 1))
 
    mrr = np.array(mrr).mean()
    print(f'Test MRR: {mrr}')
 
    return round(mrr, 4)

def average_precision_at_k(k, class_correct):
    # return average precision at k.
    # more examples: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    # and: https://www.kaggle.com/c/avito-prohibited-content#evaluation
    # class_correct is a list with the binary correct label ordered by confidence level.
    score = 0.0
    hits = 0.0
    for i in range(k):
        if class_correct[i] == 1:
            hits += 1.0
        score += hits / (i + 1.0)
    score /= k
    return score

def calc_mean_average_precision_k(df, k):
    grouped_test = df.groupby('Drug1_ID')
    average_precision = 0.0
    for (_, smile_group) in tqdm(grouped_test, leave=False):
        smile_group = smile_group.sort_values('prediction', ascending=False)
        try:
            y_true = np.array(smile_group.label.tolist())
            y_pred = np.array(smile_group['class'].tolist())
            correct_preds = np.logical_and(y_true, y_pred).astype(np.int32)
            average_precision += average_precision_at_k(k, correct_preds)
        except Exception:
            pass
    average_precision = round(average_precision / df["Drug1_ID"].nunique(), 4)
    print(f'Mean Average Precision@{k} {average_precision}', end=' ')

    return average_precision

def save_metrics(results, path):
    metrics = pd.read_csv(f'{path}/Metrics.csv')
    metrics = metrics.append(pd.Series(results), ignore_index=True)
    metrics.to_csv(f'{path}/Metrics.csv', index=False)

def calc_metrics(path: str, dataset_type: str, df: pd.DataFrame=None) -> None:
    if df is None:
        preds = pd.read_csv(f'{path}/{dataset_type}.csv')
        df = preds.copy()

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

    top_k = list(range(1, 6)) + [10, 20, 50, 100, 200]

    df = df.sort_values('prediction', ascending=False)
    df['class'] = df.prediction.progress_apply(lambda x: 1 if x > 0.5 else 0)

    for k in tqdm(top_k):
        results[f"Mean Average Precision@{k}"] = calc_mean_average_precision_k(df, k)

    print()

    results["MRR"] = calc_mrr(df)

    save_metrics(results, path)