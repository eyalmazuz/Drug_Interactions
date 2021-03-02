from unittest import TestCase

from drug_interactions.reader.dal import Drug, DrugBank

class DrugTestCase(TestCase):

    def setUp(self):
        self.drug1 = Drug(**{'id_': '1', 'interactions': [('3', 'bar')]})
        self.drug2 = Drug(**{'id_': '2', 'interactions': [('3', 'baz')]})
        self.drug3 = Drug(**{'id_': '3', 'interactions': [('1', 'bar'), ('2', 'baz')]})

    
    def testInteractsWith(self):
        self.assertTrue(self.drug1.interacts_with(self.drug3))
        self.assertFalse(self.drug2.interacts_with(self.drug1))


    def testInteractionType(self):
        self.assertEqual('bar', self.drug3.interaction_type(self.drug1))
        self.assertIsNone(self.drug1.interaction_type(self.drug2))


class DrugBankTestCase(TestCase):

    def setUp(self):
        self.drug1 = Drug(**{'id_': '1', 'interactions': {('3', 'bar')}})
        self.drug2 = Drug(**{'id_': '2', 'interactions': {('3', 'baz'), ('5', 'goo')}})
        self.drug3 = Drug(**{'id_': '3', 'interactions': {('1', 'bar'), ('2', 'baz')}})
        self.drug4 = Drug(**{'id_': '4', 'interactions': {('1', 'bar')}})

        self.drugBank = DrugBank('12', [self.drug1, self.drug2, self.drug3, self.drug4])

        self.drug2_cleaned = Drug(**{'id_': '2', 'interactions': {('3', 'baz')}})

        self.drugBankRemoved = DrugBank('12', [self.drug1, self.drug2_cleaned, self.drug3])

    
    def testHasSymmetricInteraction(self):
        self.assertTrue(self.drugBank.has_symmetric_interaction(self.drug1))
        self.assertFalse(self.drugBank.has_symmetric_interaction(self.drug4))


    def testRemoveInvalidDrugsAndInteractions(self):
        self.drugBank.remove_invalid_drugs_and_interactions(['1', '2', '3'])
        self.assertTrue(self.drugBank == self.drugBankRemoved)