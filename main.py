import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from typing import List, Tuple, Dict
import sys
print(sys.path)
import random 
import numpy as np
import matplotlib.pyplot as plt

from drug_interactions.reader.reader import DrugReader
from drug_interactions.reader.preprocessor import DrugPreprocessor
from drug_interactions.datasets.dataset_builder import get_dataset, DatasetTypes
from drug_interactions.training.train import Trainer
from drug_interactions.models.model_builder import get_model, get_config

def main():
    reader = DrugReader('./data/DrugBankReleases')

    old_drug_bank, new_drug_bank = reader.get_drug_data('5.1.3', '5.1.6')

    print(f'Old drug bank num of drugs: {len(old_drug_bank.drugs)}')
    print(f'New drug bank num of drugs: {len(new_drug_bank.drugs)}')

    print(f'Num of old drug bank interactions: {sum(map(len, [drug.interactions for drug in old_drug_bank.drugs]))}')
    print(f'Num of old drug bank interactions: {sum(map(len, [drug.interactions for drug in new_drug_bank.drugs]))}')

    smiles_length = [len(drug.smiles) for drug in new_drug_bank.drugs if drug.smiles]

    print(f'max drug smile length: {max(smiles_length)}')

    dataset_type = DatasetTypes.DEEP_SMILES

    train_dataset, validation_dataset, test_dataset, metadata = get_dataset(dataset_type, old_drug_bank,
                                                                            new_drug_bank, neg_pos_ratio=1.0, validation_size=0.2,
                                                                            atom_size=300, atom_info=21, struct_info=21, vector_size=50)

    model = get_model(dataset_type, **metadata, use_mean_vector=True)

    trainer = Trainer()

    trainer.train(model, train_dataset, validation_dataset, epochs=3, batch_size=1024, buffer_size=50000)

    trainer.predict(model, test_dataset, buffer_size=50000)

if __name__ == "__main__":
    main()
