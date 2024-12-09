import pandas as pd
import pickle
import wikipedia
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
from scipy.stats import entropy
from itertools import combinations
import ast


# Download necessary NLTK data for tokenization
# nltk.download('punkt')

# 1. Preprocessing Helper Function
def preprocess_documents(documents):
	"""Ensure all documents are valid strings and handle invalid data."""
	if not isinstance(documents, list):
		raise ValueError("Input documents must be a list.")
	return [doc if isinstance(doc, str) else "" for doc in documents]


# 2. Lexical Diversity Calculation
def calculate_lexical_diversity(documents):
	"""
	Calculate the lexical diversity of a collection of documents.

	Args:
		documents (list of str): The list of documents (strings) to be analyzed.

	Returns:
		float: The ratio of unique words to total words in the documents.
	"""
	total_words = sum(len(nltk.word_tokenize(doc)) for doc in documents if doc.strip())
	unique_words = len(set(word.lower() for doc in documents for word in nltk.word_tokenize(doc) if doc.strip()))
	return unique_words / total_words if total_words > 0 else 0


# 3. Semantic-Based Diversity Calculation
def calculate_semantic_diversity(embeddings):
	if len(embeddings) < 2:
		return 0
	cosine_sim = cosine_similarity(embeddings)
	avg_similarity = np.mean(cosine_sim[np.triu_indices_from(cosine_sim, k=1)])
	return 1 - avg_similarity


# 4. Category-Based Diversity Calculation
def calculate_category_diversity(articles):
	"""
	Calculate diversity based on the intersection and coverage of categories in the dataset.

	Args:
		articles_str (list of str): List of articles, each represented by a string of categories.

	Returns:
		dict: A dictionary with 'Coverage', 'Overlap', and 'Category Diversity Score'.
	"""
	# articles = []
	# for i in articles_str:
	#    articles.append(ast.literal_eval(i))

	# Coverage: Average number of categories per article
	avg_coverage = np.mean([len(categories) for categories in articles])

	# Pairwise Overlap Calculation: Measures the overlap of categories between pairs of articles
	overlap_sum = 0
	article_pairs = list(combinations(articles, 2))
	for cat1, cat2 in article_pairs:
		set1, set2 = set(cat1), set(cat2)
		intersection = len(set1 & set2)
		union = len(set1 | set2)
		overlap_sum += intersection / union if union else 0

	# Global Diversity: Measures how diverse the dataset is based on category overlap
	max_possible_overlap = len(article_pairs)
	diversity_score = 1 - (overlap_sum / max_possible_overlap) if max_possible_overlap > 0 else 0
	return {
		'Average Coverage': avg_coverage,
		'Category Overlap Sum': overlap_sum,
		'Category Diversity Score': diversity_score
	}


# 4. Combine Metrics into a Diversity Score Calculation
def calculate_diversity_score(submission, weights=None):
	"""
	Calculate an overall diversity score by combining lexical, semantic, and category-based diversity.

	Args:
		submission (pd.DataFrame): The submission data containing content and categories.
		weights (dict, optional): Weights for lexical, semantic, and category diversities.

	Returns:
		dict: A dictionary with the individual diversity scores and the overall diversity score.
	"""
	documents = get_dataset_column(submission, 'content')
	categories = get_dataset_column(submission, 'categories')
	embeddings = get_dataset_column(submission, 'embeddings')
	if weights is None:
		weights = {'lexical': 0.3, 'semantic': 0.4, 'category': 0.3}

	# Lexical Diversity: Calculated based on word diversity in the documents
	try:
		lexical_diversity = calculate_lexical_diversity(documents)
	except Exception as e:
		print(f"Error in lexical diversity calculation: {e}")
		lexical_diversity = 0

	# Semantic Diversity: Extracted from the 'similarity_score' of the submission
	try:
		semantic_diversity = calculate_semantic_diversity(embeddings)
	except Exception as e:
		print(f"Error in semantic diversity calculation: {e}")
		semantic_diversity = 0

	# Category Diversity: Calculated based on the diversity of categories across the dataset
	try:
		category_diversity = calculate_category_diversity(categories)['Category Diversity Score']
	except Exception as e:
		print(f"Error in semantic diversity calculation: {e}")
		category_diversity = 0

	# Calculate overall diversity score as a weighted sum of individual scores
	diversity_score = (
			weights['lexical'] * lexical_diversity +
			weights['semantic'] * semantic_diversity +
			weights['category'] * category_diversity
	)

	return {
		'Lexical Diversity': lexical_diversity,
		'Semantic Diversity': semantic_diversity,
		'Category Diversity': category_diversity,
		'Overall Diversity Score': diversity_score
	}


def get_dataset_column(dataset, columns_name):
	column = []
	for i in dataset:
		column.append(i[columns_name])
	return column


# 5. WikiRank Score Calculation
def get_wikirank_score(dataset, wikirank_df):
	"""
	Calculate the mean WikiRank score for a given dataset, ensuring all titles are present in the WikiRank dataset.

	Args:
		dataset (pd.DataFrame): The dataset containing a 'title' column.
		wikirank_df (pd.DataFrame): The DataFrame containing 'page_name' and 'wikirank_quality'.

	Returns:
		float: The mean WikiRank score for the dataset.

	Raises:
		ValueError: If any titles in the dataset are not found in wikirank_df['page_name'].
	"""
	dataset = pd.DataFrame({'title': get_dataset_column(dataset, 'title')})
	# Check if all titles in the dataset are present in the WikiRank dataset
	missing_titles = set(dataset['title']) - set(wikirank_df['page_name'])
	if missing_titles:
		raise ParticipantVisibleError(
			f"The following titles are missing from wikirank_df['page_name']: {missing_titles}")

	# Merge the datasets to calculate the mean WikiRank score
	merged_df = dataset.merge(wikirank_df, left_on='title', right_on='page_name', how='inner')

	# Extract the WikiRank quality scores and calculate the mean score
	scores = merged_df['wikirank_quality']
	return scores.mean()


class WikipediaAPI:
	def __init__(self, page_request_limit=6500,
				 wikirank_datasets_with_quality_scores_en_tsv='/kaggle/input/wikirank-datasets-with-quality-scores/en.tsv'):
		self.wikirank_df = pd.read_csv(wikirank_datasets_with_quality_scores_en_tsv, sep='\t')
		self.legal_pages = self.wikirank_df['page_name'].tolist()
		self.page_request_limit = page_request_limit
		self.list_of_known_pages = []
		self.page_requests_used = 0
		self.fetched_pages = []
		self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Initialize embedding model
		self.dataset = []

	def _increment_request(self):
		if self.page_requests_used >= self.page_request_limit:
			raise ValueError("API request limit exceeded. No further requests are allowed.")
		self.page_requests_used += 1

	def _check_legal_request(self, page_name):
		if page_name not in self.list_of_known_pages:
			raise ValueError(f"You are applying illegal request: {page_name} is not known to you")
		if page_name not in self.legal_pages:
			self.page_requests_used -= 1
			raise ValueError(f"Page {page_name} is not in the list of accessed pages. You cannot retrieve its data.")

	def search_pages(self, query):
		"""
		Searches for Wikipedia pages by query, returning a list of page names.
		Increments the API request count.

		Args:
			query (str): The search query.
			max_results (int): Maximum number of results to return.

		Returns:
			list: A list of Wikipedia page names matching the query.
		"""
		max_results = 10
		self._increment_request()
		try:
			page_names = wikipedia.search(query, results=max_results)
			self.list_of_known_pages.extend(page_names)
			return page_names
		except Exception as e:
			print(f"Search failed for query '{query}': {e}")
			return []

	def fetch_page(self, page_name):
		self._increment_request()
		self._check_legal_request(page_name)
		try:
			page = wikipedia.page(page_name)
			page_info = {
				'title': page.title,
				'content': page.content,
				'url': page.url,
				'links': page.links
			}
			self.fetched_pages.append(page)  # Save page information in the list
			self.list_of_known_pages.extend(page.links)
			return page_info
		except Exception as e:
			print(f"Failed to fetch page '{page_name}': {e}")
			return None

	def save_page(self, page_name):
		page = next((page for page in self.fetched_pages if page.title == page_name), None)
		if not page:
			raise ValueError(f"Page '{page_name}' not found in fetched pages.")

		self.dataset.append({
			'title': page.title,
			'content': page.content,
			'url': page.url,
			'links': page.links,
			'categories': page.categories,
		})

		print(f"Data of {page_name} is recorded.")
		return self.dataset

	def Calculate_embeddings(self):
		for i in range(len(self.dataset)):
			content = self.dataset[i]['content']
			self.dataset[i]['embeddings'] = self.model.encode(content)
		print("Embeddings calculated")

	def save_dataset(self, pkl_path, scores_csv_path):
		self.Calculate_embeddings()
		# Save dataset as a pickle file
		with open(pkl_path, 'wb') as f:
			pickle.dump(self.dataset, f)

		print(f"Datasets saved as .pkl file at: {pkl_path}")

		# Calculate scores and save to CSV
		diversity_score = calculate_diversity_score(self.dataset)
		wikirank_score = get_wikirank_score(self.dataset, self.wikirank_df)
		final_score = (wikirank_score + 100 * diversity_score['Overall Diversity Score']) / 2

		scores = {
			"Dataset Size": len(self.dataset),
			"WikiRank Score": wikirank_score,
			"Diversity Score": diversity_score['Overall Diversity Score'],
			"Final Score": final_score
		}
		scores_df = pd.DataFrame([scores])
		scores_df.reset_index(inplace=True)
		scores_df.rename(columns={'index': 'id'}, inplace=True)
		scores_df.to_csv(scores_csv_path, index=False)
		print(f"Scores saved to CSV file at: {scores_csv_path}")

	def is_legal_page(self, page_name):
		return page_name in self.legal_pages

	def get_usage_summary(self):
		return {
			"page_requests_used": self.page_requests_used,
			"page_request_limit": self.page_request_limit,
			"list_of_known_pages": self.list_of_known_pages
		}
