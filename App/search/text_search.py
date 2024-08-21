from ..utils.helper_utils import build_expression
class TextSimilaritySearch:
    def __init__(self, images_collection, products_collection, search_params):
        self.products_collection = products_collection
        self.images_collection = images_collection
        self.search_params = search_params
    
    def match_title_embedding_results_with_images(self, product_ids):
        try:
            # Check if product_ids is None or empty
            if not product_ids:
                print("No product IDs provided.")
                return None
            # Construct query expression
            query_expr = "p_id in {}".format(product_ids)
            
            # Execute the query
            image_results = self.images_collection.query(
                expr=query_expr,
                output_fields=["image_url", "p_id"]
            )
            return image_results
        except Exception as e:
            print(f"An error occurred while querying the images collection: {e}")
            return None
    
    def title_similarity_search(self, query_embedding, filters=None):
        expr = build_expression(filters) if filters else None
        title_results = self.products_collection.search(
            data=[query_embedding], 
            anns_field="title_vector", 
            param=self.search_params,
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
    # filters = {
    # 'price': {'operator': '>=', 'value': 3},
    # # 'store': {'operator': '==', 'value': 'Best Buy'},
    # 'average_rating': {'operator': '>', 'value': 3}
    # }
    # filters = {'main_category': {'operator': '==', 'value': 'All_Beauty'}, 'price': {'operator': '<=', 'value': 0}}
    expr = build_expression(filters)
    print(expr)