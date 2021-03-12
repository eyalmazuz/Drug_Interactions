import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from typing import List, Tuple, Dict
import sys
print(sys.path)
import random 
import numpy as np

from drug_interactions.reader.reader import DrugReader
from drug_interactions.reader.preprocessor import DrugPreprocessor
from drug_interactions.datasets.dataset_builder import get_dataset, DatasetTypes
from drug_interactions.training.train import Trainer
from drug_interactions.models.model_builder import get_model, get_config

def main():
    reader = DrugReader('./data/DrugBankReleases')

    old_drug_bank, new_drug_bank = reader.get_drug_data('5.1.3', '5.1.6')

    print(len(old_drug_bank.drugs))
    print(len(new_drug_bank.drugs))

    print(sum(map(len, [drug.interactions for drug in old_drug_bank.drugs])))
    print(sum(map(len, [drug.interactions for drug in new_drug_bank.drugs])))

    dataset_type = DatasetTypes.DEEP_SMILES

    train_dataset, validation_dataset, test_dataset, metadata = get_dataset(dataset_type, old_drug_bank,
                                                                            new_drug_bank, neg_pos_ratio=1.0, validation_size=0.2,
                                                                            atom_size=300, atom_info=21, struct_info=21)

    model = get_model(dataset_type, **metadata)

    trainer = Trainer()

    trainer.train(model, train_dataset, validation_dataset, epochs=3, batch_size=1024, buffer_size=100000)

    trainer.predict(model, test_dataset, mean_vector=True)

if __name__ == "__main__":
    main()

# slimes size < 50
# len(y)=500670 (train + validation)
# len(y_test)=356983
# validation_size=100134
# Test BCE: 0.6942437887191772 Test Accuracy: 0.0 Test AUC: 0.7054968476295471 tf.Tensor(
# [[139723 150651]
#  [ 11752  54857]], shape=(2, 2), dtype=int32)

# smiles size < 40
# len(y)=208284 (train + validation)
# len(y_test)=151713
# validation_size=41656
# Test BCE: 0.6033194065093994 Test Accuracy: 0.0 Test AUC: 0.7056666016578674 tf.Tensor(
# [[80308 44936]
#  [ 8376 18093]], shape=(2, 2), dtype=int32)