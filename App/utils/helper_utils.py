from PIL import Image

def process_user_image(image):
    return Image.fromarray(image).convert('RGB')

def calculate_score(text_score, image_score):
    text_weight = 0.5
    image_weight = 0.5
    return (text_score * text_weight) + (image_score * image_weight)

def get_product_ids(product_results = None, image_results = None):
    if image_results:
        return [result.entity.get('p_id') for result in image_results if result.entity.get('p_id')]
    elif product_results:
        return [result.entity.get('product_id') for result in product_results if result.entity.get('product_id')]

def sort_weighted_products(all_products):
    sorted_products = dict(sorted(all_products.items(), key=lambda item: item[1]['final_rank_score'], reverse=True))
    return sorted_products