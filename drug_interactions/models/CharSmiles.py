import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, GRU, Bidirectional

class CharSmilesConfig():

    def __init__(self, gru_layers: int=2, gru_units: int=32, gru_dropout_rate: float=0.2,
                    dropout_rate: float=0.2, num_classes: int=1, **kwargs):
        
        self.gru_layers = gru_layers
        self.gru_units = gru_units
        self.gru_dropout_rate = gru_dropout_rate
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.kwargs = kwargs

class CharSmiles(tf.keras.Model):

    def __init__(self, config: CharSmilesConfig):
        super(CharSmiles, self).__init__()

        self.grus = []
        for i in range(config.gru_layers):
            self.grus.append(Bidirectional(GRU(units=config.gru_units,
                                            dropout=config.gru_dropout_rate,
                                            return_sequences=i != config.gru_layers - 1),
                                            merge_mode='ave'))


        self.fc1 = Dense(units=64, activation='relu')
        self.dropout = Dropout(rate=config.dropout_rate)
        self.fc2 = Dense(units=config.num_classes, activation='sigmoid')

    def call(self, inputs, training, **kwargs):

        drug_a_smiles, drug_b_smiles = inputs

        drug_a_out = self.gru_forward(drug_a_smiles, training=training)
        drug_b_out = self.gru_forward(drug_b_smiles, training=training)

        concat = tf.concat([drug_a_out, drug_b_out], -1)
 
        dense = self.fc1(concat)
        dense = self.dropout(dense)

        logits = self.fc2(dense)

        return logits

    @tf.function()
    def gru_forward(self, drug_smiles, training):

        for gru in self.grus:
            drug_smiles = gru(drug_smiles, training=training)

        return drug_smiles

