from typing import List, Dict, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from drug_interactions.utils.utils import send_message

Data = Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], Dict[int, List[int]]]
TrainData = Tuple[List[Tuple[int, int]], List[int]]

tf.random.set_seed(0)
np.random.seed(0)

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
    def __init__(self, epoch_sample: bool=False, balance: bool=False):

        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_metrics = [tf.keras.metrics.BinaryCrossentropy(name='Train BCE'),
                        tf.keras.metrics.AUC(name='Train AUC')]
        
        self.val_metrics = [tf.keras.metrics.BinaryCrossentropy(name='Validation BCE'),
                        tf.keras.metrics.AUC(name='Validation AUC')]
        self.epoch_sample = epoch_sample
        self.balance = balance

    def train(self, model, train_dataset, validation_dataset, epochs: int=5, **kwargs):
        """
        Trains the model.
        
        Args:
            neg_pos_ratio: A float indicating how much to sample from the negative instances to the data.
                if None, then there's no sampling.
            epochs: Number of epochs to train the model.
            batch_size: Size of each batch for the model.
        """

        dataset_type = kwargs.pop('dataset_type')
        print('Start Model Training')

        print('started training')
        send_message(f'{dataset_type} started training')
        if self.balance:
            train_dataset.sample()
            validation_dataset.sample()

        for epoch in range(epochs):
            
            if self.epoch_sample:
                train_dataset.sample()
                validation_dataset.sample()

            for metric in self.train_metrics:
                metric.reset_states()

            for metric in self.val_metrics:
                metric.reset_states()

            for i, (inputs, labels) in enumerate(tqdm(train_dataset, leave=False)):
                preds = self.__train_step(model, inputs, labels, **kwargs)
                
                for metric in self.train_metrics:
                    try:
                        metric.update_state(y_true=labels, y_pred=preds)
                    except Exception:
                        print(preds.shape, labels.shape)
                        send_message(f'{preds.shape=}, {labels.shape=}')
                if (i + 1) % 500 == 0:
                    for metric in self.train_metrics:
                        print(f'{metric.name}: {round(metric.result().numpy(), 4)}', end=' ')
                        send_message(f'{dataset_type} Step {(i + 1)}: {metric.name}: {round(metric.result().numpy(), 4)}')
                    print()

            if hasattr(model, 'propegate_weights'):
                model.propegate_weights()
            print(f'Epoch: {epoch + 1} finished')
            send_message(f'{dataset_type} Epoch: {epoch + 1} finished')

            for _, (inputs, labels) in tqdm(enumerate(validation_dataset)):
                self.__validation_step(model, inputs, labels, **kwargs)
            
            for metric in self.val_metrics:
                print(f'{metric.name}: {round(metric.result().numpy(), 4)}', end=' ')
                send_message(f'{dataset_type} {metric.name}: {round(metric.result().numpy(), 4)}')
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
