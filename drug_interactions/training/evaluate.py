import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, roc_auc_score, log_loss, confusion_matrix, precision_recall_curve, auc
import  tensorflow as tf
from tqdm import tqdm, trange
tqdm.pandas()
from drug_interactions.utils.utils import send_message

def predict(model, test_dataset, save: bool=True, **kwargs):
    """
    Predicting on new Drugs and comparing to the values in the test matrix.

    Args:
        batch_size: size of the batch
        mean_vector: A boolean indicates if to use the untrained new drug embedding or take the average of existing drugs.
    """
    dataset_type = kwargs.pop('dataset_type')
    path = kwargs.pop('save_path')
    print('Building test dataset.')

    drugs_a, drugs_b, predictions, labels = [], [], [], []
    print('Predicting on the test dataset.')
    send_message(f'{dataset_type} Predicting on the test dataset.')
    
    for (new_drug_a, new_drug_b), (inputs, labels_batch) in tqdm(test_dataset, leave=False):
        preds = _test_step(model, inputs, **kwargs)

        drugs_a += list(new_drug_a)
        drugs_b += list(new_drug_b)
        predictions += [pred[0] for pred in preds.numpy().tolist()]
        labels += [l[0] for l in labels_batch.tolist()]
    
    results = {}
    results['dataset_type'] = dataset_type
    send_message(f'{dataset_type} Finished test set')
    df = pd.DataFrame({'Drug1_ID': drugs_a, 'Drug2_ID': drugs_b, 'label': labels, 'prediction': predictions})
    if save:
        df.to_csv(f'{path}/{dataset_type}.csv', index=False)
    
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
    send_message(f'{dataset_type} Test BCE: {bce}')
    send_message(f'{dataset_type} Test AUC: {roc_auc}')
    send_message(f'{dataset_type} Test PR-AUC: {pr_auc}')

    top_k = list(range(1, 21)) + [40, 60, 80, 100]

    df = df.sort_values('prediction', ascending=False)
    df['class'] = df.prediction.progress_apply(lambda x: 1 if x > 0.5 else 0)

    for k in top_k:
        y_true = np.array(df.label.tolist())
        y_pred = np.array(df['class'].tolist())
        precision_k = round(precision_score(y_true=y_true[:k], y_pred=y_pred[:k], zero_division=0), 4)
        print(f'Precision@{k}: {precision_k}')
        send_message(f'{dataset_type} Precision@{k}: {precision_k}')
        results[f'Precision@{k}'] = precision_k

    grouped_test = df.groupby('Drug1_ID')

    for k in trange(1, 6):
        average_precision = 0.0
        for (_, smile_group) in tqdm(grouped_test, leave=False):
            smile_group = smile_group.sort_values('prediction', ascending=False)
            try:
                y_true = np.array(smile_group.label.tolist())
                y_pred = np.array(smile_group['class'].tolist())
                average_precision += precision_score(y_true=y_true[:k], y_pred=y_pred[:k], zero_division=0)
            except Exception:
                pass
        average_precision = round(average_precision / df["Drug2_ID"].nunique(), 4)
        print(f'Average Precision@{k} {average_precision}', end=' ')
        send_message(f'{dataset_type} Average Precision@{k} {average_precision}')
        results[f"Average Precision@{k}"] = average_precision

    print()

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
    
    mrr = round(np.array(mrr).mean(), 4)
    print(f'Test MRR: {mrr}')
    send_message(f'{dataset_type} Test MRR: {mrr}')
    results["MRR"] = mrr

    if save:
        metrics = pd.read_csv(f'{path}/Metrics.csv')
        metrics = metrics.append(pd.Series(results), ignore_index=True)
        metrics.to_csv(f'{path}/Metrics.csv', index=False)

    send_message(f'Finished {dataset_type}')
    print(confusion_matrix(y_true=df.label.tolist(), y_pred=df['class'].tolist()))

@tf.function()
def _test_step(model, inputs: tf.Tensor, **kwargs) -> None:
    """
    Single model test step.
    after predicting on a single batch, we update the training metrics for the model.

    Args:
        drug_a_batch: A tensorflow's Tensor shape: [batch_size, 1] containing drug ids.
        drug_b_batch: A tensorflow's Tensor shape: [batch_size, 1] containing drug ids.
        labels: A tensorflow's Tensor shape: [batch_size] containing binary labels.
        mean_vector: A boolean indicates if to use the untrained new drug embedding or take the average of existing drugs.
    """
    predictions = model(inputs, training=False, **kwargs)
    
    return predictions
