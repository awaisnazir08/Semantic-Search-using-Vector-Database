import os
import numpy as np
import pandas as pd
import time
import requests
import json
import torch
import clip
import glob
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
import gradio as gr

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
    param=search_params,
    limit=20,
    expr=None,
    output_fields=["title", 'features', 'price', 'product_id', 'description', 'main_category', 'store', 'categories'],
    consistency_level="Strong")
    return title_results[0]

def match_title_embedding_results_with_images(product_ids):
    query_expr = "p_id in {}".format(product_ids)
    
    image_results = images_collection.query(
    expr=query_expr,
    output_fields=["image_url", "p_id"]
    )
    return image_results

def combine_results_retrieved_by_title_similarity(title_results, image_results):
    product_details = {}
    for result in title_results:
        product_id = result.entity.get('product_id')
        score = result.distance
        if product_id is not None:
            product_details[product_id] = {
                'title': result.entity.get('title'),
                'scores':score,
                'description': result.entity.get('description'),
                'price': result.entity.get('price'),
                'main_category': result.entity.get('main_category'),
                'store': result.entity.get('store'),
                'categories': result.entity.get('categories')
            }
    
    product_images = {}
    for image in image_results:
        product_id = image.get('p_id')
        image_url = image.get('image_url')
        
        if product_id not in product_images:
            product_images[product_id] = []
        
        product_images[product_id].append(image_url)
    
    combined_product_info = {}
    for product_id, details in product_details.items():
        combined_product_info[product_id] = details
        combined_product_info[product_id]['image_urls'] = product_images.get(product_id, [])
    return combined_product_info




def image_similarity_search(images_collection, query_embedding, search_params):
    image_results = images_collection.search(
    data=[query_embedding], 
    anns_field="image_vector", 
    param=search_params,
    limit=100,
    expr=None,
    output_fields=['p_id','image_url'],
    consistency_level="Strong",
    return_score=True
    )
    return image_results[0]

def match_image_embedding_results_with_products(product_ids):    
    query_expr = "product_id in {}".format(product_ids)
    
    product_results = products_collection.query(
        expr=query_expr,
        output_fields=["title", "description", "price", "main_category", "store", "categories"]
    )
    return product_results

def get_all_images(product_ids):
    image_query_expr = "p_id in {}".format(product_ids)
    all_images_of_matched_products = images_collection.query(
        expr=image_query_expr,
        output_fields=['image_id', 'image_url', 'p_id']
    )
    return all_images_of_matched_products

def combine_results_retrieved_by_image_similarity(product_results, image_results, all_images_of_matched_products):
    product_details = {result.get('product_id'): {
        'title': result.get('title'),
        'description': result.get('description'),
        'price': result.get('price'),
        'main_category': result.get('main_category'),
        'store': result.get('store'),
        'categories': result.get('categories'),
    } for result in product_results}
    
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
    
    for remaining_image in all_images_of_matched_products:
        if remaining_image['image_url'] not in combined_product_info[remaining_image['p_id']]['image_urls']:
            combined_product_info[remaining_image['p_id']]['image_urls'].append(remaining_image['image_url'])
            combined_product_info[remaining_image['p_id']]['scores'].append(0)
    return combined_product_info

def get_product_ids(product_results = None, image_results = None):
    if image_results:
        return [result.entity.get('p_id') for result in image_results if result.entity.get('p_id')]
    elif product_results:
        return [result.entity.get('product_id') for result in product_results if result.entity.get('product_id')]

def process_user_image(image):
    # image = Image.open(image).convert('RGB')
    return Image.fromarray(image).convert('RGB')

def calculate_score(text_score, image_score):
    text_weight = 0.6
    image_weight = 0.4
    return (text_score * text_weight) + (image_score * image_weight)

def sort_weighted_products(all_products):
    sorted_products = dict(sorted(all_products.items(), key=lambda item: item[1]['final_rank_score'], reverse=True))
    return sorted_products

def weighted_products(text_results, image_results):
    all_products = {}
    products_by_both_similarity = set()
    for id, product in text_results.items():
        if id in image_results:
            products_by_both_similarity.add(id)
            max_im_score = max(image_results[id]['scores'])
            score = calculate_score(product['scores'], max_im_score)
            all_products[id] = product
            all_products[id]['final_rank_score'] = score
        else:
            score = calculate_score(product['scores'], 0)
            all_products[id] = product
            all_products[id]['final_rank_score'] = score
    
    for id, product in image_results.items():
        if id not in products_by_both_similarity:
            score = calculate_score(0, max(product['scores']))
            all_products[id] = product
            all_products[id]['final_rank_score'] = score
    return all_products

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, preprocess, tokenizer = load_clip(clip_model='open_clip:ViT-B-16')

host = "192.168.1.101"
port = 19530

client = connections.connect("default", host=host, port=port)

print("Is Milvus connected:", connections.has_connection("default"))

print("Collections:", utility.list_collections())

db.using_database("Products")

products_collection = Collection('products')
images_collection = Collection('images')

index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 256}
}

products_collection.create_index(field_name="title_vector", index_params=index_params)
products_collection.load()

images_collection.create_index(field_name="image_vector", index_params=index_params)
images_collection.load()

search_params = {
    "metric_type": "COSINE", 
    "offset": 0, 
    "ignore_growing": False, 
    "params": {"nprobe": 20}
}

def search_by_text(text):
    query_embedding = get_text_query_embedding(text)
    title_results = title_similarity_search(products_collection, query_embedding, search_params)
    product_ids = get_product_ids(product_results=title_results)
    image_results = match_title_embedding_results_with_images(product_ids)
    results = combine_results_retrieved_by_title_similarity(title_results, image_results)
    return results

def search_by_image(image):
    image = process_user_image(image)
    query_embedding = generate_image_embeddings(image)
    image_results = image_similarity_search(images_collection, query_embedding, search_params)
    product_ids = get_product_ids(image_results=image_results)
    product_results = match_image_embedding_results_with_products(product_ids)
    all_images_of_matched_products = get_all_images(product_ids)
    results = combine_results_retrieved_by_image_similarity(product_results, image_results, all_images_of_matched_products)
    return results

def weighted_search(text, image):
    if text and text !='':
        query_embedding = get_text_query_embedding(text)
    elif image is not None:
        image = process_user_image(image)
        query_embedding = generate_image_embeddings(image)
    
    title_results = title_similarity_search(products_collection, query_embedding, search_params)
    product_ids = get_product_ids(product_results=title_results)
    image_results = match_title_embedding_results_with_images(product_ids)
    text_results = combine_results_retrieved_by_title_similarity(title_results, image_results)
    
    image_results = image_similarity_search(images_collection, query_embedding, search_params)
    product_ids = get_product_ids(image_results=image_results)
    product_results = match_image_embedding_results_with_products(product_ids)
    all_images_of_matched_products = get_all_images(product_ids)
    im_results = combine_results_retrieved_by_image_similarity(product_results, image_results, all_images_of_matched_products)
    all_products = weighted_products(text_results, im_results)
    all_products = sort_weighted_products(all_products)
    return all_products

def format_results(results):
    formatted_results = []
    for product_id, details in results.items():
        formatted_result = {
            "Title": details['title'],
            "Description": details['description'],
            "Price": details['price'],
            "Main Category": details['main_category'],
            "Store": details['store'],
            "Categories": ", ".join(details['categories']),
            "Images": details['image_urls'],
            "Score": details['scores']
        }
        if 'final_rank_score' in details:
            formatted_result['Final Rank Score'] = details['final_rank_score']
        
        formatted_results.append(formatted_result)
        
    return formatted_results


def text_search(query):
    results = search_by_text(query)
    return format_results(results)

def image_search(image):
    results = search_by_image(image)
    return format_results(results)

def text_image_search(text = None, image = None):
    # print(f'text: {text}')
    # print(f'image: {image}')
    results = weighted_search(text, image)
    return format_results(results)

text_interface = gr.Interface(
    fn=text_search,
    inputs=gr.components.Textbox(label="Search by Text"),
    outputs=gr.components.JSON(label="Results")
)

image_interface = gr.Interface(
    fn=image_search,
    inputs=gr.components.Image(label="Search by Image"),
    outputs=gr.components.JSON(label="Results")
)

text_image_interface = gr.Interface(
    fn = text_image_search,
    inputs=[
        gr.components.Textbox(label='Search by Text'),
        gr.components.Image(label='Search by Image')
    ],
    # inputs=gr.components.Textbox(label="Search by Text"),
    outputs = gr.components.JSON(label = 'Results')
)

iface = gr.TabbedInterface(
    interface_list=[text_interface, image_interface, text_image_interface],
    tab_names=["Text Search", "Image Search", 'Combined_weighted_search']
)

iface.launch()
