from ..embeddings.generate_embeddings import Embeddings
from ..search.text_search import TextSimilaritySearch
from ..search.image_search import ImageSimilaritySearch
from ..utils.helper_utils import get_product_ids, format_results, process_user_image, sort_weighted_products
from ..combine_results.combine import ProductsCombiner

class HandleSearch:
    def __init__(self, products_collection, images_collection, search_params, model, preprocess, tokenizer, device):
        self.products_collection = products_collection
        self.images_collection = images_collection
        self.search_params = search_params
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = device
        self.embeddings = Embeddings(model, preprocess, tokenizer, device)
        self.title_search = TextSimilaritySearch(self.images_collection, self.products_collection, self.search_params)
        self.image_search = ImageSimilaritySearch(self.images_collection, self.products_collection, self.search_params)
        self.combine_products = ProductsCombiner()
    
    def search_by_text(self, text, filters = None):
        query_embedding = self.embeddings.generate_text_embeddings(text)
        title_results = self.title_search.title_similarity_search(query_embedding, filters) #filters
        if not title_results:
            return None
        product_ids = get_product_ids(product_results=title_results)
        image_results = self.title_search.match_title_embedding_results_with_images(product_ids)
        results = self.combine_products.combine_results_retrieved_by_title_similarity(title_results, image_results)
        return format_results(results)

    def search_by_image(self, image, filters = None):
        image = process_user_image(image)
        query_embedding = self.embeddings.generate_image_embeddings(image)
        filtered_product_ids = None
        if filters:
            filtered_product_ids = self.image_search.get_filtered_product_ids(filters) # filters
            if not filtered_product_ids:
                return None
            filtered_product_ids = get_product_ids(filtered_products=filtered_product_ids)
        image_results = self.image_search.image_similarity_search(query_embedding, product_ids=filtered_product_ids)
        if not image_results:
            return None
        product_ids = get_product_ids(image_results=image_results)
        product_results = self.image_search.match_image_embedding_results_with_products(product_ids)
        all_images_of_matched_products = self.image_search.get_all_images(product_ids)
        results = self.combine_products.combine_results_retrieved_by_image_similarity(product_results, image_results, all_images_of_matched_products)
        return format_results(results)

    def weighted_search(self, text = None, image = None, filters = None):
        flag = [0, 0]
        
        if text and text !='':
            query_embedding = self.embeddings.generate_text_embeddings(text)
        elif image is not None:
            image = process_user_image(image)
            query_embedding = self.embeddings.generate_image_embeddings(image)
        
        title_results = self.title_search.title_similarity_search(query_embedding, filters)  # filters
        if title_results:
            # title matched results
            textual_product_ids = get_product_ids(product_results=title_results)
            image_results = self.title_search.match_title_embedding_results_with_images(textual_product_ids)
            text_results = self.combine_products.combine_results_retrieved_by_title_similarity(title_results, image_results)
        else:
            flag[0] = -1   # no title similarity results
        
        filtered_product_ids = None
        if filters:
            filtered_product_ids = self.image_search.get_filtered_product_ids(filters) #filters
            if not filtered_product_ids:
                flag[1] = -1   # no matched image results due to filters
            filtered_product_ids = get_product_ids(filtered_products=filtered_product_ids)
            
        image_results = self.image_search.image_similarity_search(query_embedding, product_ids=filtered_product_ids)
        
        if not image_results:
            flag[1] = -1   # no matched image results due to similarity
        
        if flag[0] == -1 and flag[1] == -1:
            return None
        elif flag[1] == -1 and flag[0] == 0:
            return format_results(text_results)
        else:
            product_ids = get_product_ids(image_results=image_results)
            product_results = self.image_search.match_image_embedding_results_with_products(product_ids)
            all_images_of_matched_products = self.image_search.get_all_images(product_ids)
            im_results = self.combine_products.combine_results_retrieved_by_image_similarity(product_results, image_results, all_images_of_matched_products)
        
        if flag[0] == -1 and flag[1] == 0:
            return format_results(im_results)
        
        all_products = self.combine_products.combine_weighted_products(text_results, im_results)
        all_products = sort_weighted_products(all_products)
        return format_results(all_products)


# sample filters:
# filters = {
# 'price': {'operator': '>=', 'value': 3},
# # 'store': {'operator': '==', 'value': 'Best_j'},
# 'average_rating': {'operator': '>', 'value': 3}
# }