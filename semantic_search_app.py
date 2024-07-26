import os
import numpy as np
import pandas as pd
import time
import requests
import json
import torch
import clip
import glob
import pymilvus
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    db
)
from all_clip import load_clip



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
    with torch.no_grad():
        image_embeddings = model.encode_image(img).float()
        image_embeddings /= image_embeddings.norm(dim = -1, keepdim = True)
    return image_embeddings.cpu().to(torch.float32).numpy()


def results_by_title(products_collection, query_embedding, search_params):
    results = products_collection.search(
        data=[query_embedding], 
        anns_field="title_vector",
        param=search_params,
        limit=10,
        expr=None,
        # set the names of the fields you want to 
        # retrieve from the search result.
        output_fields=['title','price', 'average_rating', 'features', 'description', 'categories', 'store', 'main_category'],
        consistency_level="Strong"
    )
    return results

def results_by_image(images_collection, query_embedding, search_params):
    results = images_collection.search(
    data=[query_embedding], 
    anns_field="image_vector",
    param=search_params,
    limit=10,
    expr=None,
    output_fields=['p_id', 'image_url'],
    consistency_level="Strong"
    )
    pid_urls = []
    for result in results[0]:
        p_id = result.entity.get('p_id')
        url = result.entity.get('image_url')
        pid_urls.append(p_id, url)
    return pid_urls

def match_product_id_with_products(p_id_list):
    matching_results  = products_collection.query(expr=f'product_id in {p_id_list}', output_fields=['title','price', 'average_rating', 'features', 'description', 'categories', 'store', 'main_category'])
    return matching_results

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, preprocess, tokenizer = load_clip(clip_model='open_clip:ViT-B-16')

client = connections.connect("default", host="localhost", port="19530")

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


sample_query = 'Casio Watch for Men'
query_embedding = get_text_query_embedding(sample_query)

products = results_by_title(query_embedding, search_params)

