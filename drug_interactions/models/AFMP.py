import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dropout, Dense

class AFMPConfig():

    def __init__(self, embedding_size: int=256, dropout_rate: float=0.3,
                    propegation_factor: float=0.4, num_classes: int=1, use_mean_vector: bool=False,
                    **kwargs):

        super().__init__()
        old_drug_bank = kwargs['old_drug_bank']
        new_drug_bank = kwargs['new_drug_bank']
        self.num_drugs = len(set(old_drug_bank.id_to_drug.keys()) | set(new_drug_bank.id_to_drug.keys()))
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.propegation_factor = propegation_factor
        self.num_classes = num_classes
        self.use_mean_vector = use_mean_vector
        self.kwargs = kwargs

class AFMP(tf.keras.Model):

    def __init__(self, config: AFMPConfig):
        super(AFMP, self).__init__()

        self.drug_embedding = Embedding(input_dim=config.num_drugs+1, output_dim=config.embedding_size, name='Embedding')
        self.bias_embedding = Embedding(input_dim=config.num_drugs+1, output_dim=1, name='Bias')

        self.dropout = Dropout(config.dropout_rate, name='Dropout')
        self.dense = Dense(units=config.num_classes, activation='sigmoid')

        self.drug_graph = self.build_drug_graph(config.kwargs['old_drug_bank'], config.kwargs['new_drug_bank'])
        self.propegation_factor = config.propegation_factor
        self.use_mean_vector = config.use_mean_vector

        self.old_drug_bank = config.kwargs['old_drug_bank']
        self.new_drug_bank = config.kwargs['new_drug_bank']

    def call(self, inputs, training=False):

        drug_a, drug_b = inputs

        if self.use_mean_vector:
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

                neighbors_weights = ((1 / len(neighbors)) * neighbors_weights).mean(axis=0)

                drug_weights = drug_weights * (1 - self.propegation_factor) + self.propegation_factor * neighbors_weights

                new_weights[drug_idx, :] = drug_weights

        self.drug_embedding.set_weights([new_weights])

    def get_mean_vector(self):
        weights = self.drug_embedding.get_weights()[0]
        old_drug_ids = list(self.drug_graph.keys())

        old_drug_embs = weights[old_drug_ids, :]

        return old_drug_embs.mean(axis=0)

    def build_drug_graph(self, old_drug_bank, new_drug_bank):
        train_drug_ids = set(old_drug_bank.id_to_drug.keys()) 
        test_drug_ids = set(new_drug_bank.id_to_drug.keys())  

        self.sorted_drug_ids = sorted(list(train_drug_ids | test_drug_ids))

        drug_graph = {}
        for drug in old_drug_bank.drugs:
            drug_index = self.sorted_drug_ids.index(drug.id_)
            drug_graph[drug_index] = [self.sorted_drug_ids.index(drug_id) for drug_id, _ in drug.interactions]

        return drug_graph
