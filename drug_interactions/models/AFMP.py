from typing import Dict, List, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Add, Dropout, Multiply, Conv2D, BatchNormalization, AveragePooling2D, \
                                Concatenate, Flatten, Dense, InputLayer, GRU, Bidirectional, GlobalMaxPool2D


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
