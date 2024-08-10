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
        product_ids = get_product_ids(product_results=title_results)
        image_results = self.title_search.match_title_embedding_results_with_images(product_ids)
        results = self.combine_products.combine_results_retrieved_by_title_similarity(title_results, image_results)
        return format_results(results)

    def search_by_image(self, image, filters = None):
        image = process_user_image(image)
        query_embedding = self.embeddings.generate_image_embeddings(image)
        filtered_product_ids = self.image_search.get_filtered_product_ids(filters) # filters
        filtered_product_ids = get_product_ids(filtered_products=filtered_product_ids)
        image_results = self.image_search.image_similarity_search(query_embedding, product_ids=filtered_product_ids)
        product_ids = get_product_ids(image_results=image_results)
        product_results = self.image_search.match_image_embedding_results_with_products(product_ids)
        all_images_of_matched_products = self.image_search.get_all_images(product_ids)
        results = self.combine_products.combine_results_retrieved_by_image_similarity(product_results, image_results, all_images_of_matched_products)
        return format_results(results)

    def weighted_search(self, text = None, image = None, filters = None):
        if text and text !='':
            query_embedding = self.embeddings.generate_text_embeddings(text)
        elif image is not None:
            image = process_user_image(image)
            query_embedding = self.embeddings.generate_image_embeddings(image)
        
        title_results = self.title_search.title_similarity_search(query_embedding, filters)  # filters
        product_ids = get_product_ids(product_results=title_results)
        image_results = self.title_search.match_title_embedding_results_with_images(product_ids)
        text_results = self.combine_products.combine_results_retrieved_by_title_similarity(title_results, image_results)
        
        filtered_product_ids = self.image_search.get_filtered_product_ids(filters) #filters
        filtered_product_ids = get_product_ids(filtered_products=filtered_product_ids)
        image_results = self.image_search.image_similarity_search(query_embedding, product_ids=filtered_product_ids)
        product_ids = get_product_ids(image_results=image_results)
        product_results = self.image_search.match_image_embedding_results_with_products(product_ids)
        all_images_of_matched_products = self.image_search.get_all_images(product_ids)
        im_results = self.combine_products.combine_results_retrieved_by_image_similarity(product_results, image_results, all_images_of_matched_products)
        
        all_products = self.combine_products.combine_weighted_products(text_results, im_results)
        all_products = sort_weighted_products(all_products)
        return format_results(all_products)