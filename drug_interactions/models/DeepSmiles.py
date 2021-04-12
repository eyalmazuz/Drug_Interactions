import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Add, Dropout, Multiply, Conv2D, BatchNormalization, AveragePooling2D, \
                                Concatenate, Flatten, Dense, InputLayer, GRU, Bidirectional, GlobalMaxPool2D

class DeepSmilesConfig():

    def __init__(self, gru_layers: int=2, gru_units: int=32, gru_dropout_rate: float=0.3,
                    dropout_rate: float=0.3, filters: int=32, num_classes: int=1, **kwargs):
        
        super().__init__()
        self.gru_layers = gru_layers
        self.gru_units = gru_units
        self.gru_dropout_rate = gru_dropout_rate
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.kwargs = kwargs


class DeepSmiles(tf.keras.Model):

    def __init__(self, config):
        super(DeepSmiles, self).__init__()

        self.atom_info = config.kwargs['atom_info']

        self.struct_info = config.kwargs['struct_info']

        # GRU Part
        self.grus = []
        for i in range(config.gru_layers):
            self.grus.append(Bidirectional(GRU(units=config.gru_units,
                                            dropout=config.gru_dropout_rate,
                                            return_sequences=i != config.gru_layers - 1),
                                            merge_mode='ave'))

        # CNN part
        self.conv1 = Conv2D(filters=config.filters, kernel_size=(3, self.atom_info + self.struct_info), strides=1, padding='valid')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters=config.filters, kernel_size=(3, 1), strides=1, padding='valid')
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        self.fcnn3 = Dense(units=64)
        
        
        # FFN part
        self.fc1 = Dense(units=64, activation='relu')
        self.dropout = Dropout(rate=config.dropout_rate)


        self.fc2 = Dense(units=config.num_classes, activation='sigmoid')

    def call(self, inputs, training, **kwargs):

        drug_a, drug_b = inputs

        drug_a_cnn, drug_a_smiles = drug_a
        drug_b_cnn, drug_b_smiles = drug_b

        drug_a_gru_out = self.gru_forward(drug_a_smiles, training=training)
        drug_a_cnn_out = self.cnn_forward(drug_a_cnn, training=training)

        drug_b_gru_out = self.gru_forward(drug_b_smiles, training=training)
        drug_b_cnn_out = self.cnn_forward(drug_b_cnn, training=training)

        drug_a_concat = tf.concat([drug_a_gru_out, drug_a_cnn_out], -1)
        drug_b_concat = tf.concat([drug_b_gru_out, drug_b_cnn_out], -1)

        drug_a_out = self.ffn(drug_a_concat, training=training)
        drug_b_out = self.ffn(drug_b_concat, training=training)

        
        concat = tf.concat([drug_a_out, drug_b_out], -1)
        
        logits = self.fc2(concat)

        return logits

    @tf.function
    def gru_forward(self, drug_smiles, training):

        for gru in self.grus:
            drug_smiles = gru(drug_smiles, training=training)

        return drug_smiles

    @tf.function
    def cnn_forward(self, drug_cnn, training):
        
        drug_cnn = tf.pad(drug_cnn, [[0, 0],[1, 1], [0, 0], [0, 0]], "CONSTANT")
        drug_cnn = tf.nn.leaky_relu(self.bn1(self.conv1(drug_cnn), training=training))
        drug_cnn = tf.pad(drug_cnn, [[0, 0],[1, 1], [0, 0], [0, 0]], "CONSTANT")
        drug_cnn = tf.nn.avg_pool2d(drug_cnn, ksize=(5, 1), strides=1, padding='VALID')
        drug_cnn = tf.pad(drug_cnn, [[0, 0],[1, 1], [0, 0], [0, 0]], "CONSTANT")
        drug_cnn = tf.nn.leaky_relu(self.bn2(self.conv2(drug_cnn), training=training))
        drug_cnn = tf.pad(drug_cnn, [[0, 0],[2, 2], [0, 0], [0, 0]], "CONSTANT")
        drug_cnn = tf.nn.avg_pool2d(drug_cnn, ksize=(5, 1), strides=1, padding='VALID')
        drug_cnn = GlobalMaxPool2D()(self.dropout(drug_cnn, training=training))
        drug_cnn = tf.nn.leaky_relu(self.bn3(self.fcnn3(drug_cnn), training=training))
        drug_cnn = self.dropout(drug_cnn, training=training)

        return drug_cnn

    @tf.function
    def ffn(self, drug_concat, training=False):

        dense = self.fc1(drug_concat)
        dense = self.dropout(dense, training=training)
        
        return dense
