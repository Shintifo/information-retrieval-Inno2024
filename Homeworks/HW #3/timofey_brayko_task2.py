import heapq
from typing import Dict, Tuple, List, Optional, Generator, Union, Any
import time
import torch
import numpy as np


class KDTree:
	def __init__(self, points: List[Tuple[int, np.ndarray]], dimension: int, distance_type: str = 'euclidean') -> None:
		"""
		Initializes a new KD-Tree and selects the distance metric.

		Args:
			points: A list of (index, embedding) tuples to build the tree from.
			dimension: The dimensionality of the embedding vectors.
			distance_type: The type of distance metric to use ('euclidean' or 'cosine'). Defaults to 'euclidean'.
		"""
		self.dimension: int = dimension
		self.root: Optional[Dict[str, Union[Tuple[int, np.ndarray], None, None]]] = self._build_tree(points)

		if distance_type == 'euclidean':
			self.distance_func = lambda a, b: np.linalg.norm(a - b)
		elif distance_type == 'cosine':
			self.distance_func = lambda a, b: 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
		else:
			raise ValueError("Invalid distance type. Use 'euclidean' or 'cosine'.")

	def _build_tree(self, points: List[Tuple[int, np.ndarray]], depth: int = 0) -> Optional[Dict[str, Any]]:
		"""
		Recursively builds the KD-Tree from the input points without modifying the input list.

		Args:
			points: The set of points to build the tree from.
			depth: The current depth of the recursion, used to determine which dimension to split along.

		Returns:
			A node in the tree structure, containing information about the point and its child nodes.
		"""
		if not points:
			return None

		axis = depth % self.dimension
		sorted_points = sorted(points, key=lambda x: x[1][axis])

		median_idx = len(sorted_points) // 2
		return {
			'point': sorted_points[median_idx],
			'left': self._build_tree(sorted_points[:median_idx], depth + 1),
			'right': self._build_tree(sorted_points[median_idx + 1:], depth + 1)
		}

	def insert(self, new_point: Tuple[int, np.ndarray]) -> None:
		"""
		Inserts a new point into the KD-Tree.

		Args:
			new_point: A tuple (index, embedding) to be added to the Tree.
		"""

		def _insert(node, point, depth):
			if node is None:
				return {'point': point, 'left': None, 'right': None}

			axis = depth % self.dimension
			if point[1][axis] < node['point'][1][axis]:
				node['left'] = _insert(node['left'], point, depth + 1)
			else:
				node['right'] = _insert(node['right'], point, depth + 1)
			return node

		self.root = _insert(self.root, new_point, 0)

	def find_knn(self, target: np.ndarray, k: int, include_distances: bool = True) -> List[
		Union[Tuple[float, Tuple[int, np.ndarray]], Tuple[int, np.ndarray]]]:
		"""
		Finds the k-nearest neighbors to a target point in the KD-Tree.

		Args:
			target: The query embedding.
			k: Number of nearest neighbors to look up.
			include_distances: Whether to return distances between query and neighbors. Default is True.

		Returns:
			List of k-nearest neighbors and optionally distances to those neighbors.
		"""
		max_heap = []
		self._search_knn(self.root, target, k, max_heap)
		result = [(abs(dist), p) for dist, p in max_heap]
		result.sort(key=lambda x: x[0])
		return result if include_distances else [r[1] for r in result]

	def _search_knn(self, curr_node: Optional[Dict[str, Any]],
					target_point: np.ndarray, k: int,
					max_heap: List[Tuple[float, Tuple[int, np.ndarray]]],
					depth: int = 0) -> None:
		"""
		Recursively searches the KD-Tree for the k-nearest neighbors.

		This method uses a max-heap to efficiently track the k closest points found so far.

		Args:
			curr_node: The current node being visited (dictionary with 'point', 'left', 'right').
			target_point: The query point.
			k: The number of nearest neighbors to find.
			max_heap: A max-heap (using heapq) storing (-distance, (index, point)).
			depth: Recursion depth (used for splitting dimension).
		"""
		if curr_node is None:
			return

		point, embedding = curr_node['point']
		distance = self.distance_func(target_point, embedding)

		# Push new node to the heap
		if len(max_heap) < k:
			# If heap is not full
			heapq.heappush(max_heap, (-distance, (point, embedding)))
		elif distance < -max_heap[0][0]:
			# If it found closer point than the farther
			heapq.heappushpop(max_heap, (-distance, (point, embedding)))

		axis = depth % self.dimension
		diff = target_point[axis] - embedding[axis]

		if diff <= 0:
			clsr = curr_node['left']
			frthr = curr_node['right']
		elif diff > 0:
			frthr = curr_node['left']
			clsr = curr_node['right']

		self._search_knn(clsr, target_point, k, max_heap, depth + 1)

		if abs(diff) < -max_heap[0][0] or len(max_heap) < k:
			self._search_knn(frthr, target_point, k, max_heap, depth + 1)

	def nearest_neighbor(self, target_point: np.ndarray, k: int = 5, include_distance: bool = True) -> Optional[
		List[Union[Tuple[float, Tuple[int, np.ndarray]], Tuple[int, np.ndarray]]]]:
		"""
		Finds the nearest neighbor to a target point by calling find_knn and returning the result up to k.

		Args:
			target_point: The query embedding.
			k: Number of nearest neighbors to look up.
			include_distances: Whether to return distances. Default is True.

		Returns:
			Optional list of the nearest points and optionally distances.
		"""
		return self.find_knn(target_point, k, include_distance)

	def __iter__(self) -> Generator[Tuple[int, np.ndarray], None, None]:
		"""
		Iterates through all stored embeddings with their indices.

		Returns:
			A generator yielding (index, embedding) tuples.
		"""

		def _traverse(node):
			if node is None:
				return
			yield node['point']
			yield from _traverse(node['left'])
			yield from _traverse(node['right'])

		yield from _traverse(self.root)

	def range_query(self, target: Union[np.ndarray, Tuple[int, np.ndarray]], radius: float) -> List[int]:
		"""
		Finds all points within a certain radius from the target point.

		Args:
			target: The query embedding.
			radius: The maximum allowable distance from the target point.

		Returns:
			A list of indices within the radius.
		"""
		results = []

		def _recursive_search(node, target_embedding, depth=0):
			if node is None:
				return

			point, embedding = node['point']
			distance = self.distance_func(target_embedding, embedding)

			if distance <= radius:
				results.append(point)

			_recursive_search(node['left'], target_embedding, depth + 1)
			_recursive_search(node['right'], target_embedding, depth + 1)

		target_embedding = target[1] if isinstance(target, tuple) else target
		_recursive_search(self.root, target_embedding)
		return results


if __name__ == "__main__":
	pass
	# image_data = torch.load("image_features.pth").numpy()
	# text_data = torch.load("text_features.pth").numpy()
	# text_data = [[idx, embed] for idx, embed in enumerate(text_data)]
	# image_data = [(idx, img) for idx, img in enumerate(image_data)]
	#
	#
	# start = time.time()
	# # Build kdtree on text data
	# kdtree = KDTree(text_data, dimension=768, distance_type='cosine')
	# end_build = time.time()
	#
	# # Retrieve the nearest neighbors for a test point
	# result = kdtree.nearest_neighbor(text_data[110][1], k=6)
	# end_search = time.time()
	# print(f"Time to build the KDTree: {end_build - start} seconds")
	# print(f"Time to build the KDTree and Search: {end_search - start} seconds")
	#
	# indices = [idx for idx, _ in [point for _, point in result]] if isinstance(result[0], tuple) else [idx for idx, _ in
	# 																								   result]
	# print("Nearset neighbors indices", indices, "\n")
