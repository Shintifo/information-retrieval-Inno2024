import io
import pprint
from typing import List, Optional

import torch
import requests
import numpy as np
from torch import nn
from PIL import Image
import torch.nn.functional as F
from datasets import load_from_disk
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel

'''
For the Cross Modal Retrieval I decided to use Multimodal Projection Learning.
The most common and known model is CLIP model. 
The main concept is to project image and text embeddings into one shared space using Feed Forward Network.
However, we should somehow keep text and image embeddings close to each other.
Hence, during the train we treat projected embeddings both for text and image the same.
And use the Cross-Entropy Loss to reduce the distance between those embeddings.

The advantage of this approach, that it handles the any dimension sized embeddings.
Moreover, due to approximation, its one of the fastest approach.

Below, I have implemented the Projection class
responsible for projecting the given embeddings to shared space.
CLIP model used to train the projections.
Please note that implementation for notebook and .py file a bit different.
For .py file I made a few changes to start the code correclty.
But the main concept and idea is not changed.
Both for jupyter notebook and .py file you will get the same results.


For me it was enough to train model for 5 epochs approximately. 
Due to low dataset size, overfitting appears even in a gap of 10 epochs
'''


class Projection(nn.Module):
	def __init__(
			self,
			embedding_dim,
			projection_dim=1024,
			dropout=0.1
	):
		super().__init__()
		self.proj = nn.Sequential(
			nn.Linear(embedding_dim, projection_dim),
			nn.GELU(),
			nn.Linear(projection_dim, projection_dim),
			nn.Dropout(dropout),
			nn.LayerNorm(projection_dim),
		)

	def forward(self, x):
		x = self.proj(x)
		return x


class CLIP(nn.Module):
	def __init__(
			self,
			temperature=1.0,
			image_embedding=768,
			text_embedding=768,
	):
		super().__init__()
		self.image_projection = Projection(embedding_dim=image_embedding)
		self.text_projection = Projection(embedding_dim=text_embedding)
		self.temperature = temperature

	def forward(self, image_features, text_features):
		device = "cuda" if torch.cuda.is_available() else "cpu"

		# Project images and captions to shared space
		image_embeddings = self.image_projection(image_features)
		text_embeddings = self.text_projection(text_features)

		# Calculate the dot product (similarity)
		logits = image_embeddings @ text_embeddings.T * np.exp(self.temperature)

		logits = logits.to(device)
		labels = torch.arange(logits.shape[1]).to(device)

		# Calculate Loss
		texts_loss = F.cross_entropy(logits, labels, reduction='mean')
		images_loss = F.cross_entropy(logits.T, labels, reduction='mean')

		# Average
		loss = (images_loss + texts_loss) / 2.0
		return loss.mean()


def save_checkpoint(model, optimizer, epoch):
	'''
	Saves models checkpoint
	:param model: Model itself
	:param optimizer: Used optimizer
	:param epoch: Current epoch
	'''
	data = {
		"model": model.state_dict(),
		"epoch": epoch,
		"optimizer": optimizer.state_dict(),
	}
	torch.save(data, "checkpoint.pth")


def train_epoch(
		model,
		loader,
		optimizer,
):
	'''
	Trains one epoch.
	:param model: CLIP model itself
	:param loader: Train dataloader
	:param optimizer: Used optimizer
	:return: None
	'''
	model.train()
	total_loss = 0.0

	for i, (img, text) in enumerate(loader, 1):
		img = img.to(device)
		text = text.to(device)

		loss = model(img, text)

		total_loss += loss.item()
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
	print("    Train Loss:", total_loss / len(loader))


def validate(
		model,
		loader,
):
	'''
	Test on the validation set.
	:param model: Model itself
	:param loader: Validation loader
	:return: Averaged loss by batches
	'''
	model.eval()
	total_loss = 0

	for i, (img, text) in enumerate(loader, 1):
		img = img.to(device)
		text = text.to(device)

		loss = model(img, text)
		total_loss += loss.item()
	print("    Validation Loss:", total_loss / len(loader))
	return total_loss / len(loader)


def train(
		clip_model,
		image_features,
		text_features,
):
	# Split dataset
	t = int(len(image_features) * 0.9)
	train_dataset = TensorDataset(image_features[:t], text_features[:t])
	val_dataset = TensorDataset(image_features[t:], text_features[t:])

	# Create dataloaders
	batch_size = 128
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

	# Hyperparameters
	lr = 1e-4
	epochs = 10
	optimizer = torch.optim.Adam(clip_model.parameters(), lr=lr)

	best_loss = torch.inf
	best_clip = None

	for epoch in range(epochs):
		print("Epoch:", epoch)
		train_epoch(clip_model, train_loader, optimizer)
		with torch.no_grad():
			loss = validate(clip_model, val_loader)
		if loss < best_loss:
			save_checkpoint(clip_model, optimizer, epoch)
			best_loss = loss
			best_clip = clip_model

	print("Best validation loss:", best_loss)
	return best_clip


def get_image_projections(clip_model, image_features) -> torch.Tensor:
	'''
	Project image features from dataset into shared space
	:param clip_model: Model itself
	:param image_features: Image features obtained from CNN/ViT
	:return: Projected image features from dataset
	'''
	with torch.no_grad():
		image_features = image_features.to(device)
		image_embeddings = clip_model.image_projection(image_features).to(device)
	return image_embeddings


def get_text_projections(clip_model, text_features) -> torch.Tensor:
	'''
	Project text features from dataset into shared space
	:param clip_model: Model itself
	:param text_features: Text features obtained from RNN/Transformer
	:return: Projected text features from dataset
	'''
	with torch.no_grad():
		text_features = text_features.to(device)
		text_embeddings = clip_model.text_projection(text_features).to(device)
	return text_embeddings


def find_matches(query, project_query, embeddings, model, n=6):
	query_embed = project_query(model, query)  # Project query to shared space
	sim = torch.nn.functional.cosine_similarity(query_embed, embeddings)
	vals, indices = torch.topk(sim, n)
	return indices


def inference(
		clip_model,
		query: np.ndarray,
		ds,
		scope_features: np.ndarray | torch.tensor,
		mode: str,
):
	'''
	Produce the search
	:param clip_model: Model itself
	:param query: Query image or text
	:param ds: Loaded dataset
	:param scope_features: Scope through which the search will be done. Features of text of images obtained from CNN/ViT or RNN/Transformer respectively.
	:param mode: Search type: txt2img, txt2txt, img2txt, img2img
	:return:
	'''

	# For each case:
	# 	1. Project searched scope with clip model
	# 	2. Find matches query with projected features
	# 	3. Display the results

	match mode:
		case "txt2img":
			image_embeddings = get_image_projections(clip_model, scope_features)
			indices = find_matches(query, get_text_projections, image_embeddings, clip_model)
			plot_results(indices, ds)
		case "img2txt":
			text_embeddings = get_text_projections(clip_model, scope_features)
			indices = find_matches(query, get_image_projections, text_embeddings, clip_model)
			print_results(indices, ds)
		case "img2img":
			image_embeddings = get_image_projections(clip_model, scope_features)
			indices = find_matches(query, get_image_projections, image_embeddings, clip_model)
			plot_results(indices, ds)
		case "txt2txt":
			text_embeddings = get_text_projections(clip_model, scope_features)
			indices = find_matches(query, get_text_projections, text_embeddings, clip_model)
			print_results(indices, ds)
		case _:
			raise Exception("Invalid mode")


def load_image_from_url(url: str) -> np.ndarray:
	response = requests.get(url)
	if response.status_code == 200:
		img = Image.open(io.BytesIO(response.content))
		return np.array(img)


def plot_results(indices, ds):
	num_neighbors = len(indices)
	num_rows = int(num_neighbors ** 0.5)
	num_cols = int(np.ceil(num_neighbors / num_rows))

	fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

	for i, ax in enumerate(axes.flat):
		if i < num_neighbors:
			x = load_image_from_url(ds["url"][indices[i]])
			ax.imshow(x)
			ax.set_title(f"Neighbor {i + 1}")
			ax.axis("off")

	plt.tight_layout()
	plt.show()


def print_results(indices, ds):
	print("Results", indices, "\n")
	sentences_to_print = [ds["sentences"][i] for i in indices]
	pprint.pprint(sentences_to_print, indent=4)


if __name__ == "__main__":
	ds = load_from_disk("task_dataset")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Used device:", device)

	# Load precomputed embeddings:
	image_features = torch.load("image_features.pth")
	text_features = torch.load("text_features.pth")

	# Define model
	clip_model = CLIP().to(device)

	# Training
	clip_model = train(clip_model, image_features, text_features)

	# Example of inference txt2img
	print("TEXT TO IMAGE QUERY")
	test_point = 5729
	query = text_features[test_point]
	print("Query:", ds["sentences"][test_point][0])
	inference(clip_model, query, ds, scope_features=image_features, mode="txt2img")



	# Example of inference img2txt
	print("IMAGE TO TEXT QUERY")
	test_point = 5729
	query = image_features[test_point]
	plt.imshow(load_image_from_url(ds['url'][test_point]))
	plt.title("Query Image")
	plt.axis('off')
	plt.show()

	inference(clip_model, query, ds, scope_features=text_features, mode="img2txt")
