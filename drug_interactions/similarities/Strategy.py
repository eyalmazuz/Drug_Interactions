from abc import ABC, abstractmethod

class Strategy(ABC):

	def __init__(self, similar_map_path):
		pass


	@abstractmethod
	def find_similars(self, drug_id):
		pass