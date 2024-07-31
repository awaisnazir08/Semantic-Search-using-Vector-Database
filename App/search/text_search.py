def title_similarity_search(products_collection, query_embedding, search_params):
    title_results = products_collection.search(
        data=[query_embedding], 
        anns_field="title_vector", 
        param=search_params,
        limit=30,
        expr=None,
        output_fields=["title", 'features', 'price', 'product_id', 'description', 'main_category', 'store', 'categories'],
        consistency_level="Strong"
    )
    return title_results[0]


def match_title_embedding_results_with_images(images_collection, product_ids):
    query_expr = "p_id in {}".format(product_ids)
    
    image_results = images_collection.query(
    expr=query_expr,
    output_fields=["image_url", "p_id"]
    )
    return image_results