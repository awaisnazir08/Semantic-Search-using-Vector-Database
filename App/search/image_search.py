from ..utils.helper_utils import build_expression

def image_similarity_search(images_collection, query_embedding, search_params, product_ids = None):
    if product_ids:
        expr = f"p_id in [{', '.join(map(str, product_ids))}]"
    image_results = images_collection.search(
        data=[query_embedding], 
        anns_field="image_vector", 
        param=search_params,
        limit=100,
        expr=None if not product_ids else expr,
        output_fields=['p_id', 'image_url'],
        consistency_level="Strong",
        return_score=True
    )
    return image_results[0]


# def match_image_embedding_results_with_products(products_collection, product_ids):    
#     query_expr = "product_id in {}".format(product_ids)
    
#     product_results = products_collection.query(
#         expr=query_expr,
#         output_fields=["title", "description", "price", "main_category", "store", "categories", 'features', 'average_rating']
#     )
#     return product_results


def get_all_images(images_collection, product_ids):
    image_query_expr = "p_id in {}".format(product_ids)
    all_images_of_matched_products = images_collection.query(
        expr=image_query_expr,
        output_fields=['image_id', 'image_url', 'p_id']
    )
    return all_images_of_matched_products

def match_image_embedding_results_with_products(products_collection, product_ids):
    query_expr = "product_id in {}".format(product_ids)
    
    product_results = products_collection.query(
        expr=query_expr,
        output_fields=["title", "description", "price", "main_category", "store", "categories", 'features', 'average_rating', 'product_id']
    )
    return product_results

def get_filtered_product_ids(products_collection, filters = None):
    filters = {
    'price': {'operator': '>=', 'value': 3},
    # 'store': {'operator': '==', 'value': 'Best Buy'},
    'average_rating': {'operator': '>', 'value': 3}
    }
    if filters: 
        query_expr = build_expression(filters)
        product_id_results = products_collection.query(
        expr=query_expr,
        output_fields=['product_id']
    )
    else:
        return None
    return product_id_results