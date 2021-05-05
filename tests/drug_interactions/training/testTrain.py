from unittest import TestCase

import numpy as np

from drug_interactions.reader.dal import Drug, DrugBank
from drug_interactions.training.train import Trainer
from drug_interactions.datasets.dataset_builder import ColdStartDrugDataset

class ColdStartDrugDatasetTestCase(TestCase):

    def setUp(self):

        self.Olddrug1 = Drug(**{'id_': '1', 'interactions': {('3', 'bar')}})
        self.Olddrug2 = Drug(**{'id_': '2', 'interactions': {('3', 'baz')}})
        self.Olddrug3 = Drug(**{'id_': '3', 'interactions': {('1', 'bar'), ('2', 'baz')}})
  

        self.Newdrug1 = Drug(**{'id_': '1', 'interactions': {('3', 'bar'), ('4', 'bar')}})
        self.Newdrug2 = Drug(**{'id_': '2', 'interactions': {('3', 'baz'), ('5', 'bar')}})
        self.Newdrug3 = Drug(**{'id_': '3', 'interactions': {('1', 'bar'), ('2', 'baz')}})
        self.Newdrug4 = Drug(**{'id_': '4', 'interactions': {('1', 'bar'), ('6', 'goo')}})
        self.Newdrug5 = Drug(**{'id_': '5', 'interactions': {('2', 'bar')}})
        self.Newdrug6 = Drug(**{'id_': '6', 'interactions': {('4', 'goo')}})

        self.oldDrugBank = DrugBank('9', [self.Olddrug1, self.Olddrug2, self.Olddrug3])

        self.NewDrugBank = DrugBank('12', [self.Newdrug1, self.Newdrug2, self.Newdrug3, self.Newdrug4, self.Newdrug5, self.Newdrug6])

        self.dataset = ColdStartDrugDataset(self.oldDrugBank, self.NewDrugBank, neg_pos_ratio=1.0)

    # def testCreateColdStartData(self):
    #     train_matrix, test_matrix, new_drugs_idxs, drug_graph = self.dataset.create_data(self.oldDrugBank, self.NewDrugBank, 'cold_start')

    #     gt_train_matrix = np.array([[0,0,1,0,0,0],
    #                                 [0,0,1,0,0,0],
    #                                 [1,1,0,0,0,0],
    #                                 [0,0,0,0,0,0],
    #                                 [0,0,0,0,0,0],
    #                                 [0,0,0,0,0,0]])

    #     gt_test_matrix = np.array([[0,0,1,1,0,0],
    #                                 [0,0,1,0,1,0],
    #                                 [1,1,0,0,0,0],
    #                                 [1,0,0,0,0,1],
    #                                 [0,1,0,0,0,0],
    #                                 [0,0,0,1,0,0]])

    #     gt_new_drugs_idxs = [(3, 0), (3, 1), (3, 2),
    #                         (4, 0), (4, 1), (4, 2), (4, 3),
    #                         (5, 0), (5, 1), (5, 2), (5, 3), (5, 4)]

    #     gt_drug_graph = {
    #         0: [2],
    #         1: [2],
    #         2: [0, 1],
    #     }

    #     self.assertTrue(np.all(train_matrix == gt_train_matrix))
    #     self.assertTrue(np.all(test_matrix == gt_test_matrix))
    #     self.assertCountEqual(new_drugs_idxs, gt_new_drugs_idxs)
    #     self.assertCountEqual(drug_graph, gt_drug_graph)


    # def testGetPositiveSamplesTrain(self):

    #     train_matrix, _, new_drugs_idxs, _ = self.dataset.create_data()

    #     gt_positive_samples = [(2, 1), (2, 0)]
    #     gt_positive_labels = [1, 1]


    #     positive_samples, positive_labels = self.dataset.get_positive_instances(train_matrix, new_drugs_idxs)

    #     self.assertCountEqual(positive_samples, gt_positive_samples)
    #     self.assertCountEqual(positive_labels, gt_positive_labels)

    
    # def testGetPositiveSamplesTest(self):

    #     _, test_matrix, new_drugs_idxs, _ = self.dataset.create_data()

    #     gt_positive_samples = [(2, 1), (2, 0)]
    #     gt_positive_labels = [1, 1]


    #     positive_samples, positive_labels = self.dataset.get_positive_instances(test_matrix, new_drugs_idxs)

    #     self.assertCountEqual(positive_samples, gt_positive_samples)
    #     self.assertCountEqual(positive_labels, gt_positive_labels)

    
    # def testGetNegativeSamplesTrain(self):

    #     train_matrix, _, new_drugs_idxs, _ = self.dataset.create_data()

    #     gt_negative_samples = [(1, 0)]
    #     gt_negative_labels = [0]


    #     negative_samples, negative_labels = self.dataset.get_negative_instances(train_matrix, new_drugs_idxs)

    #     self.assertCountEqual(negative_samples, gt_negative_samples)
    #     self.assertCountEqual(negative_labels, gt_negative_labels)

    
    # def testGetNegativeSamplesTest(self):

    #     _, test_matrix, new_drugs_idxs, _ = self.dataset.create_data()

    #     gt_negative_samples = [(1, 0)]
    #     gt_negative_labels = [0]


    #     negative_samples, negative_labels = self.dataset.get_negative_instances(test_matrix, new_drugs_idxs)

    #     self.assertCountEqual(negative_samples, gt_negative_samples)
    #     self.assertCountEqual(negative_labels, gt_negative_labels)


    # def testBuildTestDataset(self):
    #     gt_x_test = [np.array([3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]),
    #                 np.array([0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4])]
        
    #     gt_y_test = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]

    #     x_test, y_test = self.dataset.build_test_dataset(self.dataset.new_drug_idxs)
    #     sorted_idxs = np.argsort(x_test[0], kind='stable')
    #     x_test[0] = x_test[0][sorted_idxs]
    #     x_test[1] = x_test[1][sorted_idxs]

    #     y_test_new = []
    #     for i in sorted_idxs:
    #         y_test_new.append(y_test[i])
    #     y_test = y_test_new

    #     self.assertCountEqual(y_test, gt_y_test)
    #     print(x_test)
    #     self.assertTrue(np.all(x_test[0] == gt_x_test[0]))
    #     # self.assertTrue(np.all(x_test[1] == gt_x_test[1])) # TODO FIX THIS.
