import os
import numpy as np
import pandas as pd
import time
import requests
import json
import torch
import clip
import glob
# import pymilvus
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    db,
    MilvusClient
)
import unicodedata
import open_clip
from all_clip import load_clip
from PIL import Image


IMAGES_URL_DIRECTORY = 'image_folder'
IMAGE_EMBEDDINGS_DIRECTORY = 'image_embeddings_complete'
TITLE_EMBEDDINGS_DIRECTORY = 'title_embeddings'
title_embeddings_path = 'title_embeddings/text_emb/text_emb_0.npy'
image_embeddings_path = 'image_embeddings_complete/img_emb/img_emb_0.npy'
image_embeddings_meta_data_file = 'image_embeddings_complete/metadata/metadata_0.parquet'
title_embeddings_meta_data_file = 'title_embeddings/metadata/metadata_0.parquet'
data_path = 'D:\Datasets\Amazon\Amazon Product Dataset\meta_All_Beauty.jsonl'


def get_text_query_embedding(query):
    # Load the model and preprocess function
    text_tokens = tokenizer(query)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    text_embs = text_features.cpu().to(torch.float32).numpy()
    return text_embs[0]

def generate_image_embeddings(img):
    img = preprocess(img).unsqueeze(0).to(device) 
    with torch.no_grad():
        image_embeddings = model.encode_image(img).float()
        image_embeddings /= image_embeddings.norm(dim = -1, keepdim = True)
    return image_embeddings.cpu().to(torch.float32).numpy()[0]



def title_similarity_search(products_collection, query_embedding, search_params):
    title_results = products_collection.search(
    data=[query_embedding], 
    anns_field="title_vector", 
    # the sum of `offset` in `param` and `limit` 
    # should be less than 16384.
    param=search_params,
    limit=20,
    expr=None,
    # set the names of the fields you want to 
    # retrieve from the search result.
    output_fields=["title", 'features', 'price', 'product_id', 'description', 'main_category', 'store', 'categories'],
    consistency_level="Strong")
    return title_results[0]

def match_title_embedding_results_with_images(title_results):
    product_ids = [result.entity.get('product_id') for result in title_results]
    query_expr = "p_id in {}".format(product_ids)
    
    # Search for images
    image_results = images_collection.query(
    expr=query_expr,
    output_fields=["image_url", "p_id"]  # Adjust fields based on your schema
    )
    return image_results

def combine_results_retrieved_by_title_similarity(title_results, image_results):
    # Step 1: Access Product Details from title_results
    product_details = {}
    for result in title_results:  # Adjust according to actual structure of title_results
        product_id = result.entity.get('product_id')  # Adjust according to actual field names
        score = result.distance
        if product_id is not None:
            product_details[product_id] = {
                'title': result.entity.get('title'),
                'score':score,
                'description': result.entity.get('description'),
                'price': result.entity.get('price'),
                'main_category': result.entity.get('main_category'),
                'store': result.entity.get('store'),
                'categories': result.entity.get('categories')
            }

    # Step 2: Create a dictionary for image URLs
    product_images = {}
    for image in image_results:
        product_id = image.get('p_id')
        image_url = image.get('image_url')
        
        if product_id not in product_images:
            product_images[product_id] = []
        
        product_images[product_id].append(image_url)
    
    # Step 3: Combine Product Details and Image URLs into a Single Dictionary
    combined_product_info = {}
    for product_id, details in product_details.items():
        combined_product_info[product_id] = details
        combined_product_info[product_id]['image_urls'] = product_images.get(product_id, [])
    return combined_product_info

def image_similarity_search(images_collection, query_embedding, search_params):
    image_results = images_collection.search(
    data=[query_embedding], 
    anns_field="image_vector", 
    # the sum of `offset` in `param` and `limit` 
    # should be less than 16384.
    param=search_params,
    limit=100,
    expr=None,
    # set the names of the fields you want to 
    # retrieve from the search result.
    output_fields=['p_id','image_url'],
    consistency_level="Strong",
    return_score = True
    )
    # Flatten the results
    image_results = image_results[0]  # Access the actual result data
    return image_results

def match_image_embedding_results_with_products(image_results):
    # Extract product_ids from the image search results
    product_ids = [result.entity.get('p_id') for result in image_results if result.entity.get('p_id')]

    # Construct query expression to retrieve product information
    query_expr = "product_id in {}".format(product_ids)

    # Perform the query on the products collection
    product_results = products_collection.query(
        expr=query_expr,
        output_fields=["title", "description", "price", "main_category", "store", "categories"]
    )
    return product_results

def combine_results_retrieved_by_image_similarity(product_results, image_results):
    # Process results
    product_details = {result.get('product_id'): {
        'title': result.get('title'),
        'description': result.get('description'),
        'price': result.get('price'),
        'main_category': result.get('main_category'),
        'store': result.get('store'),
        'categories': result.get('categories'),
    } for result in product_results}
    
    # Create a dictionary to combine product details and image URLs
    combined_product_info = {}
    for image in image_results:
        product_id = image.entity.get('p_id')
        image_url = image.entity.get('image_url')
        score = image.distance
        
        if product_id in product_details:
            if product_id not in combined_product_info:
                combined_product_info[product_id] = product_details[product_id]
                combined_product_info[product_id]['image_urls'] = []
                combined_product_info[product_id]['scores'] = []
            combined_product_info[product_id]['image_urls'].append(image_url)
            combined_product_info[product_id]['scores'].append(score)
    return combined_product_info



device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, preprocess, tokenizer = load_clip(clip_model='open_clip:ViT-B-16')

host = "141.195.16.189"
port = 40238 # Mapping for 19530 (default Milvus port)

# Connect to Milvus
client = connections.connect("default", host=host, port=port)

# Check if the connection is established
print("Is Milvus connected:", connections.has_connection("default"))

# Optional: List collections to confirm the connection
from pymilvus import utility
print("Collections:", utility.list_collections())

db.using_database("Products")

products_collection = Collection('products')
images_collection = Collection('images')

index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 256}
}

products_collection.create_index(field_name="title_vector", index_params = index_params)
products_collection.load()

images_collection.create_index(field_name="image_vector", index_params = index_params)
images_collection.load()

search_params = {
    "metric_type": "COSINE", 
    "offset": 0, 
    "ignore_growing": False, 
    "params": {"nprobe": 20}
}

# Title similarity search 
# ---------------------------------------
# if button clicked for search by text, user enters text, and then run this block of code and return the results
sample_query = 'Casio Watch for Men'
query_embedding = get_text_query_embedding(sample_query)

title_results = title_similarity_search(products_collection, query_embedding, search_params)

image_results = match_title_embedding_results_with_images(title_results)

products_by_title_similarity_search = combine_results_retrieved_by_title_similarity(title_results, image_results)
# ---------------------------------------

# Image Similarity Search 
# ---------------------------------------
# if user clicks on button to upload a picture, then run this block of code and return the results
sample_image = 'sample_image.jpeg'
image = Image.open(sample_image).convert('RGB')
query_embedding = generate_image_embeddings(image)

image_results = image_similarity_search(images_collection, query_embedding, search_params)

product_results = match_image_embedding_results_with_products(image_results)

products_by_image_similarity_search = combine_results_retrieved_by_image_similarity(product_results, image_results)