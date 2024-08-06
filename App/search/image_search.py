def image_similarity_search(images_collection, query_embedding, search_params):
    image_results = images_collection.search(
        data=[query_embedding], 
        anns_field="image_vector", 
        param=search_params,
        limit=100,
        expr=None,
        output_fields=['p_id', 'image_url'],
        consistency_level="Strong",
        return_score=True
    )
    return image_results[0]


def match_image_embedding_results_with_products(products_collection, product_ids):    
    query_expr = "product_id in {}".format(product_ids)
    
    product_results = products_collection.query(
        expr=query_expr,
        output_fields=["title", "description", "price", "main_category", "store", "categories", 'features', 'average_rating']
    )
    return product_results


def get_all_images(images_collection, product_ids):
    image_query_expr = "p_id in {}".format(product_ids)
    all_images_of_matched_products = images_collection.query(
        expr=image_query_expr,
        output_fields=['image_id', 'image_url', 'p_id']
    )
    return all_images_of_matched_products

# def match_image_embedding_results_with_products(products_collection, product_ids, filters=None):
#     product_ids_expr = "product_id in [{}]".format(", ".join(map(str, product_ids)))
    
#     if filters:
#         filters_expr = build_expression(filters)
#         query_expr = f"{product_ids_expr} and {filters_expr}"
#     else:
#         query_expr = product_ids_expr
    
#     product_results = products_collection.query(
#         expr=query_expr,
#         output_fields=["title", "description", "price", "main_category", "store", "categories", 'features', 'average_rating']
#     )
#     return product_results