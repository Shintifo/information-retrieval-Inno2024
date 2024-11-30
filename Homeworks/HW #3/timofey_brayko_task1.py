import requests
import io

import torch
import numpy as np
from PIL import Image
from datasets import load_from_disk
from transformers import AutoModel, AutoImageProcessor
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional, Union, Generator, Dict, Any




def load_image_from_url(url: str) -> np.ndarray:
	"""Loads an image from a given URL and converts it to an RGB NumPy array.

	Args:
	  url: The URL of the image to load.

	Returns:
	  A NumPy array representing the image in RGB format.
	"""
	response = requests.get(url)
	if response.status_code == 200:
		img = Image.open(io.BytesIO(response.content))
		return np.array(img)



def encode_image(img):
	"""Encodes a given image URL into an embedding vector.

	Args:
	  image:  The URL of the image to encode.

	Returns:
	   A NumPy array representing the image embedding.
	"""

	img = load_image_from_url(img)

	with torch.no_grad():
		inputs = processor(images=img, return_tensors="pt")
		inputs = inputs.to("cuda")
		outputs = image_model(**inputs).last_hidden_state[:, 0, :]

	return outputs.squeeze(0).detach().cpu().numpy()



def encode_text(captions: List[List[str]]) -> List[np.ndarray]:
	# Take only the first caption fot the encoding
	c = []
	for cap in captions:
		c.append(cap[0])
	captions = c
	# Produce the embeddings
	embeds = text_model.encode(captions)
	return embeds



def process_images(indexation=True):
	'''
	Image embedding loop. Process images through chosen model
	:param indexation: Whether return array of indexed embeddings or pure list of embeddings.
	:return: Obtained indexes
	'''
	image_model.eval()
	image_data = []
	for u in ds["url"]:
		image_data.append(encode_image(u))
	# Index the data
	if indexation:
		image_data = [(idx, img) for idx, img in enumerate(image_data)]
	return image_data


def process_texts(indexation=True):
	'''
	Text embedding loop. Process captions through chosen model
	:param indexation: Whether return array of indexed embeddings or pure list of embeddings.
	:return: Obtained indexes
	'''
	text_data = encode_text(ds['sentences'])
	if indexation:
		text_data = [[idx, embed] for idx, embed in enumerate(text_data)]
	return text_data


if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	ds = load_from_disk("task_dataset")

	dimensions = 768
	text_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=dimensions).to(device)

	processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
	image_model = AutoModel.from_pretrained("google/vit-base-patch16-224").to(device)

	image_data = process_images()
	text_data = process_texts()


