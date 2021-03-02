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
from drug_interactions.models.model import get_model, ModelTypes

def main():
    reader = DrugReader('./data/DrugBankReleases')

    old_drug_bank, new_drug_bank = reader.get_drug_data('5.1.3', '5.1.6')

    print(len(old_drug_bank.drugs))
    print(len(new_drug_bank.drugs))

    print(sum(map(len, [drug.interactions for drug in old_drug_bank.drugs])))
    print(sum(map(len, [drug.interactions for drug in new_drug_bank.drugs])))

    train_dataset, validation_dataset, test_dataset, metadata = get_dataset(DatasetTypes.COLD_START, old_drug_bank,
                                                                            new_drug_bank, neg_pos_ratio=1.0, validation_size=0.2)

    metadata = {**metadata, **{"embedding_size": 128, "dropout_rate": 0.3, "num_classes": 1, "propegation_factor": 0.4}}

    model = get_model(ModelTypes.AFMP, metadata)

    trainer = Trainer()

    trainer.train(model, train_dataset, validation_dataset, epochs=3, batch_size=1024, validation_size=int(metadata['data_size']*0.2), buffer_size=1000000)

    trainer.predict(model, test_dataset, mean_vector=True)

if __name__ == "__main__":
    main()
