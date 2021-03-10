from typing import List, Dict, Any, Tuple
from itertools import product
import random

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix

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
                        tf.keras.metrics.AUC(name='Train AUC')]
        
        self.val_metrics = [tf.keras.metrics.BinaryCrossentropy(name='Validation BCE'),
                        tf.keras.metrics.AUC(name='Validation AUC')]

    def train(self, model, train_dataset, validation_dataset, epochs: int=5, batch_size: int=1024, validation_size: int=1000, buffer_size=1000000):
        """
        Trains the model.
        
        Args:
            neg_pos_ratio: A float indicating how much to sample from the negative instances to the data.
                if None, then there's no sampling.
            epochs: Number of epochs to train the model.
            batch_size: Size of each batch for the model.
            validation_size: How much from all the data to take for the validation set.
        """
        print('Start Model Training')
        print(f'{validation_size=}')
        train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
        validation_dataset = validation_dataset.shuffle(buffer_size).batch(batch_size)

        print('started training')
        for epoch in range(epochs):

            for metric in self.train_metrics:
                metric.reset_states()

            for metric in self.val_metrics:
                metric.reset_states()

            for i, (inputs, labels) in tqdm(enumerate(train_dataset)):
                self.__train_step(model, inputs, labels)

                if (i + 1) % 200 == 0:
                    for metric in self.train_metrics:
                        print(f'{metric.name}: {metric.result().numpy()}', end=' ')
                    print()
            # model.propegate_weights()
            print(f'Epoch: {epoch + 1} finished')

            for _, (inputs, labels) in tqdm(enumerate(validation_dataset)):
                self.__validation_step(model, inputs, labels)
            
            for metric in self.val_metrics:
                print(f'{metric.name}: {metric.result().numpy()}', end=' ')
            print('Done Validation.')

        print('Finished training')

    @tf.function()
    def __validation_step(self, model, inputs: tf.Tensor, labels: tf.Tensor) -> None:
        """
        Single model validaiton step.
        after predicting on a single batch, we update the training metrics for the model.

        Args:
            drug_a_batch: A tensorflow's Tensor shape: [batch_size, 1] containing drug ids.
            drug_b_batch: A tensorflow's Tensor shape: [batch_size, 1] containing drug ids.
            labels: A tensorflow's Tensor shape: [batch_size] containing binary labels.
        """
        predictions = model(inputs, training=False)
        for metric in self.val_metrics:
            metric.update_state(y_true=labels, y_pred=predictions)
    
    @tf.function()
    def __train_step(self, model, inputs: tf.Tensor, labels: tf.Tensor) -> None:
        """
        Single model train step.
        after predicting on a single batch, we update the training metrics for the model.

        Args:
            drug_a_batch: A tensorflow's Tensor shape: [batch_size, 1] containing drug ids.
            drug_b_batch: A tensorflow's Tensor shape: [batch_size, 1] containing drug ids.
            labels: A tensorflow's Tensor shape: [batch_size] containing binary labels.
        """
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = self.loss_fn(y_true=labels, y_pred=predictions)        
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        for metric in self.train_metrics:
            metric.update_state(y_true=labels, y_pred=predictions)


    def predict(self, model, test_dataset, batch_size: int=1024, buffer_size: int=1*10**6, mean_vector: bool=False):
        """
        Predicting on new Drugs and comparing to the values in the test matrix.
        
        Args:
            batch_size: size of the batch
            mean_vector: A boolean indicates if to use the untrained new drug embedding or take the average of existing drugs.
        """
        print('Building test dataset.')
        test_dataset = test_dataset.shuffle(buffer_size).batch(batch_size)
        test_metrics = [tf.keras.metrics.BinaryCrossentropy(name='Test BCE'),
                        tf.keras.metrics.AUC(name='Test AUC')]

        for metric in test_metrics:
                metric.reset_states()

        predictions, labels = [], []
        print('Predicting on the test dataset.')
        for _, (inputs, labels_batch) in tqdm(enumerate(test_dataset)):
            preds = self.__test_step(model, inputs, labels_batch, test_metrics, mean_vector)

            predictions += [x[0] for x in preds.numpy().tolist()]
            labels += labels_batch.numpy().tolist()
        
        binary_predictions = [1 if x > 0.5 else 0 for x in predictions]

        print('Done predicting.')
        for metric in test_metrics:
            print(f'{metric.name}: {metric.result().numpy()}', end=' ')
        
        print(tf.math.confusion_matrix(labels, binary_predictions))

    @tf.function()
    def __test_step(self, model, inputs: tf.Tensor,
                        labels: tf.Tensor, metrics: List[tf.keras.metrics.Metric], mean_vector: bool=False) -> None:
        """
        Single model test step.
        after predicting on a single batch, we update the training metrics for the model.

        Args:
            drug_a_batch: A tensorflow's Tensor shape: [batch_size, 1] containing drug ids.
            drug_b_batch: A tensorflow's Tensor shape: [batch_size, 1] containing drug ids.
            labels: A tensorflow's Tensor shape: [batch_size] containing binary labels.
            mean_vector: A boolean indicates if to use the untrained new drug embedding or take the average of existing drugs.
        """
        predictions = model(inputs, training=False, mean_vector=mean_vector)
        for metric in metrics:
            metric.update_state(y_true=labels, y_pred=predictions)
        
        return predictions
    