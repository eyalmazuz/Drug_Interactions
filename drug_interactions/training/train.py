from typing import List, Dict, Any, Tuple
from itertools import product
import random

import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score, confusion_matrix

from drug_interactions.utils import webhook_url, send_message
from drug_interactions.reader.dal import DrugBank
from drug_interactions.reader.preprocessor import DrugPreprocessor


Data = Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], Dict[int, List[int]]]
TrainData = Tuple[List[Tuple[int, int]], List[int]]

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class Trainer():

    """
    A class that manages all the data from training models and evaluating.


    Attributes:
        train_bank: A drug bank data used for the model training.
        test_bank: A drug bank data used for the model evaluation.
        data_type: String indicating which type of data to create.
        propegation_factor: A float of the amount of propegation for the model training.
    """
    def __init__(self, propegation_factor: float=0.4):

        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_metrics = [tf.keras.metrics.BinaryCrossentropy(name='Train BCE'),
                        tf.keras.metrics.AUC(name='Train AUC')] + [
                        tf.keras.metrics.Precision(top_k=i, name=f'Train Precision@{i}') for i in [1, 20, 40 ,60, 80, 100]
                        ]
        
        self.val_metrics = [tf.keras.metrics.BinaryCrossentropy(name='Validation BCE'),
                        tf.keras.metrics.AUC(name='Validation AUC')] + [
                        tf.keras.metrics.Precision(top_k=i, name=f'Validation Precision@{i}') for i in [1, 20, 40 ,60, 80, 100]
                        ]

    def train(self, model, train_dataset, validation_dataset, epochs: int=5,
                    batch_size: int=1024, buffer_size=1000, **kwargs):
        """
        Trains the model.
        
        Args:
            neg_pos_ratio: A float indicating how much to sample from the negative instances to the data.
                if None, then there's no sampling.
            epochs: Number of epochs to train the model.
            batch_size: Size of each batch for the model.
        """
        print('Start Model Training')
        train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(3)
        validation_dataset = validation_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(3)

        print('started training')
        for epoch in range(epochs):

            for metric in self.train_metrics:
                metric.reset_states()

            for metric in self.val_metrics:
                metric.reset_states()

            for i, (inputs, labels) in enumerate(tqdm(train_dataset, leave=False)):
                preds = self.__train_step(model, inputs, labels, **kwargs)
                
                for metric in self.train_metrics:
                    try:
                        metric.update_state(y_true=labels, y_pred=preds)
                    except:
                        print(preds.shape, labels.shape)
                        send_message(f'{preds.shape=}, {labels.shape=}')
                if (i + 1) % 500 == 0:
                    for metric in self.train_metrics:
                        print(f'{metric.name}: {round(metric.result().numpy(), 4)}', end=' ')
                        send_message(f'Step {(i + 1)}: {metric.name}: {round(metric.result().numpy(), 4)}')
                    print()

            if hasattr(model, 'propegate_weights'):
                model.propegate_weights()
            print(f'Epoch: {epoch + 1} finished')
            send_message(f'Epoch: {epoch + 1} finished')

            for _, (inputs, labels) in tqdm(enumerate(validation_dataset)):
                self.__validation_step(model, inputs, labels, **kwargs)
            
            for metric in self.val_metrics:
                print(f'{metric.name}: {round(metric.result().numpy(), 4)}', end=' ')
                send_message(f'{metric.name}: {round(metric.result().numpy(), 4)}')
                print()
            print('Done Validation.')

        print('Finished training')

    @tf.function()
    def __validation_step(self, model, inputs: tf.Tensor, labels: tf.Tensor, **kwargs) -> None:
        """
        Single model validaiton step.
        after predicting on a single batch, we update the training metrics for the model.

        Args:
            drug_a_batch: A tensorflow's Tensor shape: [batch_size, 1] containing drug ids.
            drug_b_batch: A tensorflow's Tensor shape: [batch_size, 1] containing drug ids.
            labels: A tensorflow's Tensor shape: [batch_size] containing binary labels.
        """
        predictions = model(inputs, training=False, **kwargs)

        for metric in self.val_metrics:
                metric.update_state(y_true=labels, y_pred=predictions)

    @tf.function()
    def __train_step(self, model, inputs: tf.Tensor, labels: tf.Tensor, **kwargs) -> None:
        """
        Single model train step.
        after predicting on a single batch, we update the training metrics for the model.

        Args:
            drug_a_batch: A tensorflow's Tensor shape: [batch_size, 1] containing drug ids.
            drug_b_batch: A tensorflow's Tensor shape: [batch_size, 1] containing drug ids.
            labels: A tensorflow's Tensor shape: [batch_size] containing binary labels.
        """
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True, **kwargs)
            loss = self.loss_fn(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return predictions


    def predict(self, model, test_dataset, batch_size: int=1024, buffer_size: int=1*10**5, **kwargs):
        """
        Predicting on new Drugs and comparing to the values in the test matrix.

        Args:
            batch_size: size of the batch
            mean_vector: A boolean indicates if to use the untrained new drug embedding or take the average of existing drugs.
        """
        print('Building test dataset.')
        test_dataset = test_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(3)
        test_metrics = [tf.keras.metrics.BinaryCrossentropy(name='Test BCE'),
                        tf.keras.metrics.AUC(name='Test AUC')]  + [
                        tf.keras.metrics.Precision(top_k=i, name=f'Test Precision@{i}') for i in [1, 20, 40 ,60, 80, 100]
                        ]

        for metric in test_metrics:
                metric.reset_states()

        predictions, labels = [], []
        drug_predictions = {}
        print('Predicting on the test dataset.')
        send_message('Predicting on the test dataset.')
        
        for i, (new_drug_idxs, (inputs, labels_batch)) in enumerate(tqdm(test_dataset, leave=False)):
            preds = self._test_step(model, inputs, labels_batch, test_metrics, **kwargs)
            
            for new_drug, prediction, label in zip(new_drug_idxs, preds.numpy(), labels_batch.numpy()):
                new_drug = new_drug.numpy()
                if new_drug not in drug_predictions:
                    drug_predictions[new_drug] = {'predictions': [], 'labels': []}
                drug_predictions[new_drug]['predictions'].append(prediction[0])
                drug_predictions[new_drug]['labels'].append(label)

            predictions += [x[0] for x in preds.numpy().tolist()]
            labels += labels_batch.numpy().tolist()
        
        print('Done predicting.')
        for metric in test_metrics:
            print(f'{metric.name}: {round(metric.result().numpy(), 4)}', end=' ')
            send_message(f'{metric.name}: {round(metric.result().numpy(), 4)}')
            print()

        binary_predictions = [1 if x > 0.5 else 0 for x in predictions]
        
        print(tf.math.confusion_matrix(labels, binary_predictions))
        send_message(f'{tf.math.confusion_matrix(labels, binary_predictions)}')
        
        len_new_drugs = len(drug_predictions)
        print(len_new_drugs)
        
        for i in trange(5):
            average_precision = 0.0
            for outputs in tqdm(drug_predictions.values(), leave=False):
                map_ = tf.keras.metrics.Precision(top_k=i+1, name=f'Test Average Precision@{i + 1}')
                try:
                    map_.update_state(y_true=outputs['labels'], y_pred=outputs['predictions'])
                    average_precision += map_.result().numpy()
                except:
                    pass
            send_message(f'{map_.name}: {round(average_precision / len_new_drugs, 4)}')
            print(f'{map_.name}: {round(average_precision / len_new_drugs, 4)}', end=' ')

    @tf.function()
    def _test_step(self, model, inputs: tf.Tensor,
                        labels: tf.Tensor, metrics: List[tf.keras.metrics.Metric], **kwargs) -> None:
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
        for metric in metrics:
            metric.update_state(y_true=labels, y_pred=predictions)
        
        return predictions
    