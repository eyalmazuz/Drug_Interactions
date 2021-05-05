import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, roc_auc_score, log_loss, confusion_matrix
import  tensorflow as tf
from tqdm import tqdm, trange
tqdm.pandas()
from drug_interactions.utils import send_message

def predict(model, test_dataset, **kwargs):
    """
    Predicting on new Drugs and comparing to the values in the test matrix.

    Args:
        batch_size: size of the batch
        mean_vector: A boolean indicates if to use the untrained new drug embedding or take the average of existing drugs.
    """
    dataset_type = kwargs.pop('dataset_type')
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
    
    send_message(f'{dataset_type} Finished test set')
    df = pd.DataFrame({'smile_a': drugs_a, 'smile_b': drugs_b, 'label': labels, 'prediction': predictions})
    print(df.head())
 
    bce = log_loss(y_true=df.label.tolist(), y_pred=df.prediction.tolist())
    auc = roc_auc_score(y_true=df.label.tolist(), y_score=df.prediction.tolist())
    print(f'Test BCE: {round(bce, 4)}')
    print(f'Test AUC: {round(auc, 4)}')
    send_message(f'{dataset_type} Test BCE: {round(bce, 4)}')
    send_message(f'{dataset_type} Test AUC: {round(auc, 4)}')

    top_k = [1, 20, 40, 60, 80, 100]

    df = df.sort_values('prediction', ascending=False)
    df['class'] = df.prediction.progress_apply(lambda x: 1 if x > 0.5 else 0)

    for k in top_k:
        y_true = np.array(df.label.tolist())
        y_pred = np.array(df['class'].tolist())
        print(f'Precision@{k}: {round(precision_score(y_true=y_true[:k], y_pred=y_pred[:k], zero_division=0), 4)}')
        send_message(f'{dataset_type} Precision@{k}: {round(precision_score(y_true=y_true[:k], y_pred=y_pred[:k], zero_division=0), 4)}')

    grouped_test = df.groupby('smile_a')

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
        print(f'Average Precision@{k} {round(average_precision / df["smile_a"].nunique(), 4)}', end=' ')
        send_message(f'{dataset_type} Average Precision@{k} {round(average_precision / df["smile_a"].nunique(), 4)}')
    print()
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
