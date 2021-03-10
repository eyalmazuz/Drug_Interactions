import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Add, Dropout, Multiply, Conv2D, BatchNormalization, AveragePooling2D, \
                                Concatenate, Flatten, Dense, InputLayer, GRU, Bidirectional, GlobalMaxPool2D


class CharSmiles(tf.keras.Model):

    def __init__(self, metadata):
        super(CharSmiles, self).__init__()

        self.atomsize = metadata['atomsize']

        self.grus = []
        for i in range(metadata['gru_layers']):
            self.grus.append(Bidirectional(GRU(units=metadata['gru_units'],
                                            dropout=metadata['gru_dropout_rate'],
                                            return_sequences=i != metadata['gru_layers'] - 1),
                                            merge_mode='ave'))


        self.fc1 = Dense(units=64, activation='relu')
        self.dropout = Dropout(rate=metadata['dropout_rate'])
        self.fc2 = Dense(units=metadata['num_classes'], activation='sigmoid')

    def call(self, inputs, training, **kwargs):

        drug_a_smiles, drug_b_smiles = inputs


        drug_a_out = self.gru_forward(drug_a_smiles, training=training)
        drug_b_out = self.gru_forward(drug_b_smiles, training=training)


        concat = tf.concat([drug_a_out, drug_b_out], -1)
        
        dense = self.fc1(concat)
        dense = self.dropout(dense)

        logits = self.fc2(dense)

        return logits

    def gru_forward(self, drug_smiles, training):

        for gru in self.grus:
            drug_smiles = gru(drug_smiles, training=training)

        return drug_smiles

