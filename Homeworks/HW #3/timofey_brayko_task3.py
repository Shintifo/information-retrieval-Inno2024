from typing import List, Tuple

import numpy as np

class LSH:
	def __init__(self, index_data: np.ndarray, bucket_size: int = 3, seed: int = 42, distance_type: str = 'cosine',
				 num_tables: int = 5):
		"""
		Initialize LSH with data, bucket size, random seed, and distance type.

		:param index_data: Array of tuples where each tuple consists of an index and data point.
		:param bucket_size: Number of data points per bucket.
		:param seed: Seed for random number generator.
		:param distance_type: Type of distance metric, either 'euclidean' or 'cosine'.
		:param num_tables: Number of hash tables to use for repeated searches.
		"""
		self.indices, self.data = zip(*index_data)
		self.data = np.asarray(self.data)
		self.bucket_size = bucket_size
		self.rng = np.random.default_rng(seed)
		self.num_tables = num_tables
		self.hyperplanes = [self._generate_hyperplanes() for _ in range(num_tables)]
		self.hash_tables = [self._create_hash_table(i) for i in range(num_tables)]

		if distance_type == 'euclidean':
			self.distance_func = self._euclidean_distance
		elif distance_type == 'cosine':
			self.distance_func = self._cosine_distance
		else:
			raise ValueError("Invalid distance type. Use 'euclidean' or 'cosine'.")

	def _generate_hyperplanes(self) -> np.ndarray:
		"""
		Generate random hyperplanes for hashing based on feature dimensions and bucket size.

		:return: Array of hyperplanes for hashing data.
		"""
		feature_dim = self.data.shape[1]
		num_hyperplanes = self.bucket_size
		return self.rng.normal(size=(num_hyperplanes, feature_dim))

	def _generate_hash_key(self, points: np.ndarray, hyperplanes: np.ndarray) -> np.ndarray:
		"""
		Generate a hash key for given points based on the hyperplanes
		Remember that you need to convert the resulting binary hash into a decimal value.

		:param points: Array of data points to hash.
		:param hyperplanes: Array of hyperplanes to use for hashing.
		:return: Hash keys for the data points.
		"""
		binary_hashes = np.dot(points, hyperplanes.T) >= 0
		return binary_hashes.dot(1 << np.arange(binary_hashes.shape[1] - 1, -1, -1))

	def _query_hash_candidates(self, query: np.ndarray, repeat: int = 10) -> List[int]:
		"""
		Retrieve candidates from hash table based on query and specified repeat count.

		:param query: Query data point.
		:param repeat: Number of times to hash the query for candidate retrieval.
		:return: List of candidate indices.
		"""
		candidates = set()
		for i in range(min(repeat, self.num_tables)):
			hash_key = self._generate_hash_key(query[np.newaxis, :], self.hyperplanes[i])[0]
			if hash_key in self.hash_tables[i]:
				candidates.update(self.hash_tables[i][hash_key])
		return list(candidates)

	def _euclidean_distance(self, points: np.ndarray, query: np.ndarray) -> np.ndarray:
		return np.linalg.norm(points - query, axis=1)

	def _cosine_distance(self, points: np.ndarray, query: np.ndarray) -> np.ndarray:
		return 1 - np.dot(points, query) / (np.linalg.norm(points, axis=1) * np.linalg.norm(query))

	def _create_hash_table(self, table_index: int) -> dict:
		"""
		Create a hash table for the LSH algorithm by mapping data points to hash buckets.

		:param table_index: Index of the hash table to create.
		:return: Hash table with keys as hash values and values as lists of data indices.
		"""
		hash_table = {}
		hash_keys = self._generate_hash_key(self.data, self.hyperplanes[table_index])

		for idx, hash_key in enumerate(hash_keys):
			if hash_key not in hash_table:
				hash_table[hash_key] = []
			hash_table[hash_key].append(self.indices[idx])

		return hash_table

	def approximate_knn_search(self, query: np.ndarray, k: int = 5, repeat: int = 10) -> Tuple[
		np.ndarray, np.ndarray, np.ndarray]:
		"""
		Perform approximate K-nearest neighbor search on the query point.

		:param query: Query point for which nearest neighbors are sought.
		:param k: Number of neighbors to retrieve.
		:param repeat: Number of times to hash the query to increase candidate count.
		:return: Tuple of nearest points, their distances, and their original indices.
		"""
		candidates_indices = self._query_hash_candidates(query, repeat=repeat)
		if not candidates_indices:
			return np.array([]), np.array([]), np.array([])

		candidate_points = self.data[candidates_indices]
		distances = self.distance_func(candidate_points, query)

		nearest_indices = np.argsort(distances)[:k]
		nearest_points = candidate_points[nearest_indices]
		nearest_distances = distances[nearest_indices]
		original_indices = np.array(candidates_indices)[nearest_indices]

		return nearest_points, nearest_distances, original_indices








if __name__ == "__main__":
	pass
