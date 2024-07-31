from utils.helper_utils import calculate_score

def combine_weighted_products(text_results, image_results):
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