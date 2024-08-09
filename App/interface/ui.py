import gradio as gr
from ..search.text_search import title_similarity_search, match_title_embedding_results_with_images
from ..search.image_search import get_all_images, match_image_embedding_results_with_products, image_similarity_search, get_filtered_product_ids
from ..combine_results.combine import combine_weighted_products, combine_results_retrieved_by_image_similarity, combine_results_retrieved_by_title_similarity
from ..utils.helper_utils import process_user_image, get_product_ids, sort_weighted_products
from ..embeddings.text_embedding import get_text_query_embedding
from ..embeddings.image_embedding import generate_image_embeddings
import torch
from all_clip import load_clip
from ..model.load_model import get_model

def format_results(results):
    formatted_results = []
    for product_id, details in results.items():
        formatted_result = {
            "Title": details['title'],
            "Description": details['description'],
            'Features': details['features'],
            "Price": details['price'],
            'Average Rating': details['average_rating'],
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

def search_by_text(text, products_collection, images_collection, search_params, model, tokenizer):
    query_embedding = get_text_query_embedding(model, tokenizer, text)
    title_results = title_similarity_search(products_collection, query_embedding, search_params)
    product_ids = get_product_ids(product_results=title_results)
    image_results = match_title_embedding_results_with_images(images_collection, product_ids)
    results = combine_results_retrieved_by_title_similarity(title_results, image_results)
    return format_results(results)

def search_by_image(image, products_collection, images_collection, search_params, model, preprocess, device):
    image = process_user_image(image)
    query_embedding = generate_image_embeddings(image, model, preprocess, device)
    filtered_product_ids = get_filtered_product_ids(products_collection)
    filtered_product_ids = get_product_ids(filtered_products=filtered_product_ids)
    image_results = image_similarity_search(images_collection, query_embedding, search_params, product_ids=filtered_product_ids)
    product_ids = get_product_ids(image_results=image_results)
    product_results = match_image_embedding_results_with_products(products_collection, product_ids)
    all_images_of_matched_products = get_all_images(images_collection, product_ids)
    results = combine_results_retrieved_by_image_similarity(product_results, image_results, all_images_of_matched_products)
    return format_results(results)

def weighted_search(products_collection, images_collection, search_params, model, tokenizer, preprocess, device, text = None, image = None):
    if text and text !='':
        query_embedding = get_text_query_embedding(model, tokenizer, text)
    elif image is not None:
        image = process_user_image(image)
        query_embedding = generate_image_embeddings(image, model, preprocess, device)
    
    title_results = title_similarity_search(products_collection, query_embedding, search_params)
    product_ids = get_product_ids(product_results=title_results)
    image_results = match_title_embedding_results_with_images(images_collection, product_ids)
    text_results = combine_results_retrieved_by_title_similarity(title_results, image_results)
    
    filtered_product_ids = get_filtered_product_ids(products_collection)
    filtered_product_ids = get_product_ids(filtered_products=filtered_product_ids)
    image_results = image_similarity_search(images_collection, query_embedding, search_params, product_ids=filtered_product_ids)
    product_ids = get_product_ids(image_results=image_results)
    product_results = match_image_embedding_results_with_products(products_collection, product_ids)
    all_images_of_matched_products = get_all_images(images_collection, product_ids)
    im_results = combine_results_retrieved_by_image_similarity(product_results, image_results, all_images_of_matched_products)
    
    all_products = combine_weighted_products(text_results, im_results)
    all_products = sort_weighted_products(all_products)
    return format_results(all_products)


def launch_interface(products_collection, images_collection, search_params, device):
    
    model, preprocess, tokenizer = get_model()
    
    text_interface = gr.Interface(
        fn=lambda text: search_by_text(text, products_collection, images_collection, search_params, model, tokenizer),
        inputs=gr.components.Textbox(label="Search by Text"),
        outputs=gr.components.JSON(label="Results")
    )

    image_interface = gr.Interface(
        fn=lambda image: search_by_image(image, products_collection,  images_collection, search_params, model, preprocess, device),
        inputs=gr.components.Image(label="Search by Image"),
        outputs=gr.components.JSON(label="Results")
    )

    text_image_interface = gr.Interface(
        fn=lambda text, image: weighted_search(products_collection, images_collection, search_params, model, tokenizer, preprocess, device, text, image),
        inputs=[
            gr.components.Textbox(label='Search by Text'),
            gr.components.Image(label='Search by Image')
        ],
        outputs=gr.components.JSON(label='Results')
    )

    iface = gr.TabbedInterface(
        interface_list=[text_interface, image_interface, text_image_interface],
        tab_names=["Text Search", "Image Search", 'Combined_weighted_search']
    )

    iface.launch()