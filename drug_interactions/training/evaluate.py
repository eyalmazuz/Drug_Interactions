import numpy as np
import pandas as pd
import  tensorflow as tf
from tqdm import tqdm
tqdm.pandas()
from drug_interactions.utils.utils import send_message
from drug_interactions.utils.calc_metrics import calc_metrics

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
    
    send_message(f'{dataset_type} Finished test set')
    df = pd.DataFrame({'Drug1_ID': drugs_a, 'Drug2_ID': drugs_b, 'label': labels, 'prediction': predictions})
    if save:
        df.to_csv(f'{path}/{dataset_type}.csv', index=False)

        calc_metrics(path, dataset_type)

    send_message(f'Finished {dataset_type}')
    # print(confusion_matrix(y_true=df.label.tolist(), y_pred=df['class'].tolist()))

def predict_tta(model, test_dataset, save: bool=True, **kwargs):
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
    for (new_drug_a, new_drug_b), (inputs, labels_batch), tta_weights in tqdm(test_dataset, leave=False):
        
        preds = _test_step(model, inputs, **kwargs)
        drugs_a += list([new_drug_a])
        drugs_b += list([new_drug_b])
            
        # calc tta average
        if tta_weights:
            prediction = [pred[0] for pred in preds.numpy().tolist()]
            prediction = [np.average(prediction, weights=tta_weights)]
        else:
            prediction = [pred[0] for pred in preds.numpy().tolist()]

        predictions += prediction
        labels += labels_batch.tolist()[0]
    send_message(f'{dataset_type} Finished test set')
    print(len(drugs_a), len(drugs_b), len(labels), len(predictions))
    df = pd.DataFrame({'Drug1_ID': drugs_a, 'Drug2_ID': drugs_b, 'label': labels, 'prediction': predictions})
    if save:
        df.to_csv(f'{path}/{dataset_type}.csv', index=False)

        calc_metrics(path, dataset_type)

    send_message(f'Finished {dataset_type}')

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
