from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Add, Dropout, Multiply, Conv2D, BatchNormalization, AveragePooling2D, \
                                Concatenate, Flatten, Dense, InputLayer, GRU, Bidirectional, GlobalMaxPool2D

class BinarySmilesConfig():

    def __init__(self, num_layers: int=4, dense_units: List[int]=[512, 256, 128, 64], 
                dropout_rate: float=0.2, num_classes: int=1, **kwargs):
        
        super().__init__()
        self.num_layers = num_layers
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.kwargs = kwargs

class BinarySmiles(tf.keras.Model):

    def __init__(self, config: BinarySmilesConfig):
        super(BinarySmiles, self).__init__()

        self.dense_layers = []
        for i in range(config.num_layers):
            self.dense_layers.append(Dense(units=config.dense_units[i]))

        self.dropout = Dropout(rate=config.dropout_rate)
        self.final = Dense(units=config.num_classes, activation='sigmoid')

    @tf.function()
    def call(self, x, training, **kwargs):

        for dense in self.dense_layers:
            x = dense(x)
            x = tf.nn.relu(x)
            x = self.dropout(x)

        logits = self.final(x)

        return logits


