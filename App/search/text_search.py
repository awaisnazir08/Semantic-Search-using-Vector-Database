# def title_similarity_search(products_collection, query_embedding, search_params):
#     title_results = products_collection.search(
#         data=[query_embedding], 
#         anns_field="title_vector", 
#         param=search_params,
#         limit=30,
#         expr=None,
#         output_fields=["title", 'features', 'price', 'product_id', 'description', 'main_category', 'store', 'categories', 'average_rating'],
#         consistency_level="Strong"
#     )
#     return title_results[0]

def match_title_embedding_results_with_images(images_collection, product_ids):
    query_expr = "p_id in {}".format(product_ids)
    
    image_results = images_collection.query(
    expr=query_expr,
    output_fields=["image_url", "p_id"]
    )
    return image_results

def build_expression(filters):
    expressions = []
    for field, condition in filters.items():
        operator = condition.get("operator", "==")
        value = condition.get("value")
        
        if operator not in ["==", "!=", "<", "<=", ">", ">=", "in"]:
            raise ValueError(f"Unsupported operator: {operator}")
        
        if operator == "in" and isinstance(value, list):
            value_list = ', '.join([f'"{v}"' if isinstance(v, str) else str(v) for v in value])
            expressions.append(f'{field} in [{value_list}]')
        else:
            if isinstance(value, str):
                value = f'"{value}"'
            expressions.append(f'{field} {operator} {value}')
    return " and ".join(expressions) if expressions else None

def title_similarity_search(products_collection, query_embedding, search_params, filters=None):
    filters = {
    'price': {'operator': '>=', 'value': 3},
    # 'store': {'operator': '==', 'value': 'Best Buy'},
    'average_rating': {'operator': '>', 'value': 3}
    }
    expr = build_expression(filters) if filters else None
    title_results = products_collection.search(
        data=[query_embedding], 
        anns_field="title_vector", 
        param=search_params,
        limit=30,
        expr=expr,
        output_fields=["title", 'features', 'price', 'product_id', 'description', 'main_category', 'store', 'categories','average_rating'],
        consistency_level="Strong"
    )
    return title_results[0]

if __name__ == '__main__':
    filters = {
    'price': {'operator': '>=', 'value': 3},
    # 'store': {'operator': '==', 'value': 'Best Buy'},
    'average_rating': {'operator': '>', 'value': 3}
    }

    expr = build_expression(filters)
    print(expr)