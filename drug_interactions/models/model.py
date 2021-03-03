from enum import Enum
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from typing import Dict, List, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Add, Dropout, Multiply, Conv2D, BatchNormalization, AveragePooling2D, \
                                Concatenate, Flatten, Dense, InputLayer, GRU, Bidirectional, GlobalMaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback


class ModelTypes(Enum):
    AFMP = 1
    SMILES = 2

class AFMP(tf.keras.Model):

    def __init__(self, metadata: Dict[str, Any]):
        super(AFMP, self).__init__()

        self.drug_embedding = Embedding(input_dim=metadata['num_drugs']+1, output_dim=metadata['embedding_size'], name='Embedding')
        self.bias_embedding = Embedding(input_dim=metadata['num_drugs']+1, output_dim=1, name='Bias')

        self.dropout = Dropout(metadata['dropout_rate'], name='Dropout')
        self.dense = Dense(units=metadata['num_classes'], activation='sigmoid')

        self.drug_graph = metadata['drug_graph']
        self.propegation_factor = metadata['propegation_factor']


    def call(self, inputs, training=False, **kwargs):
        
        drug_a, drug_b = inputs

        if 'mean_vector' in kwargs and kwargs['mean_vector']:
            print('using mean vector')
            drug_a_emb = self.drug_embedding(np.array(list(self.drug_graph.keys())), training=True)
            drug_a_emb = tf.math.reduce_mean(drug_a_emb, axis=0)
            drug_a_emb = tf.repeat([drug_a_emb], drug_b.shape[0], axis=0)
        else:
            drug_a_emb = self.drug_embedding(drug_a, training=True)
        drug_a_emb = self.dropout(drug_a_emb, training=training)

        drug_b_emb = self.drug_embedding(drug_b, training=True)
        drug_b_emb = self.dropout(drug_b_emb, training=training)
        
        mult = tf.multiply(drug_a_emb, drug_b_emb)

        drug_a_bias = self.bias_embedding(drug_a, training=True)
        drug_b_bias = self.bias_embedding(drug_b, training=True)
        add = tf.add(drug_a_bias, drug_b_bias)        

        final = tf.concat([mult, add], -1)

        logits = self.dense(final, training=True)

        return logits

    def propegate_weights(self):

        print('propegating weights')
        weights = self.drug_embedding.get_weights()[0]
        new_weights = np.zeros(shape=weights.shape)

        for drug_idx, neighbors in self.drug_graph.items():

            if len(neighbors) > 0:
                drug_weights = weights[drug_idx, :]
                neighbors_weights = weights[neighbors, :]

                # total_weights += (1 / len(neighbors)) * neighbors
                neighbors_weights = ((1 / len(neighbors)) * neighbors_weights).mean(axis=0)

                drug_weights = drug_weights * (1 - self.propegation_factor) + self.propegation_factor * neighbors_weights

                new_weights[drug_idx, :] = drug_weights
        
        self.drug_embedding.set_weights([new_weights])
    
    def get_mean_vector(self):
        weights = self.drug_embedding.get_weights()[0]
        old_drug_ids = list(self.drug_graph.keys())

        old_drug_embs = weights[old_drug_ids, :]

        return old_drug_embs.mean(axis=0)


class DeepSmiles(tf.keras.Model):

    def __init__(self, metadata):
        super(DeepSmiles, self).__init__()

        self.atomsize = metadata['atomsize']
        self.vocab_size = metadata['vocab_size']

        self.len_info = metadata['len_info']

        self.grus = []
        for _ in range(metadata['gru_layers']):
            self.grus.append(Bidirectional(GRU(units=metadata['gru_units'], dropout=metadata['gru_dropout_date']), merge_mode='ave'))

        self.conv1 = Conv2D(filters=32, kernel_size=(3, 42), strides=1, padding=(1, 0))
        self.bn1 = BatchNormalization()
        self.conv2 = self.conv1 = Conv2D(filters=32, kernel_size=(3, 1), strides=1, padding=(1, 0))
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        self.dropout = Dropout(rate=metadata['dropout_rate'])
        self.fcnn3 = Dense(units=64)
        self.fc1 = Dense(units=64)
        self.final = Dense(units=1, activation='sigmoid')

    def call(self, inputs, training, **kwargs):

        drug_a, drug_b = inputs
        drug_a_cnn, drug_a_smiles = drug_a
        drug_b_cnn, drug_b_smiles = drug_b

        drug_a_out = self.pass_(drug_a_cnn, drug_a_smiles, training=training)
        drug_b_out = self.pass_(drug_b_cnn, drug_b_smiles, training=training)

        final = tf.concat([drug_a_out, drug_b_out], -1)

        logits = self.final(final)

    def pass_(self, drug_cnn, drug_smiles, training):

        for gru in self.grus:
            drug_smiles = gru(drug_smiles)

        drug_cnn = tf.nn.leaky_relu(self.bn1(self.conv1(drug_smiles), training=training))
        drug_cnn = tf.nn.avg_pool2d(drug_cnn, ksize=(5, 1), strides=1, padding=(2, 0))
        drug_cnn = tf.nn.leaky_relu(self.bn2(self.conv2(drug_cnn), training=training))
        drug_cnn = tf.nn.avg_pool2d(drug_cnn, ksize=(5, 1), strides=1, padding=(2, 0))
        drug_cnn = GlobalMaxPool2D()(self.dropout(drug_cnn, training=training))
        drug_cnn = tf.squeeze(drug_cnn, -1)
        drug_cnn = tf.squeeze(drug_cnn, -1)
        drug_cnn = tf.nn.leaky_relu(self.bn3(self.fcnn3(drug_cnn), training=training))
        drug_cnn = self.dropout(drug_cnn, training=training)

        concat = tf.concat([drug_cnn, drug_smiles], -1)

        output = tf.nn.relu(self.fc1(concat))
        output = self.dropout(output, training=training)

        return output

        


def get_model(model_type: ModelTypes, metadata:Dict[str, Any]):

    if model_type == ModelTypes.AFMP:
        return AFMP(metadata)

    if model_type == ModelTypes.SMILES:
        return DeepSmiles(metadata)