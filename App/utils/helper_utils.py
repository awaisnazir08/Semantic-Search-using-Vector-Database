from PIL import Image
import yaml

def process_user_image(image):
    return Image.fromarray(image).convert('RGB')

def calculate_score(text_score, image_score):
    text_weight = 0.5
    image_weight = 0.5
    return (text_score * text_weight) + (image_score * image_weight)

def get_product_ids(product_results = None, image_results = None, filtered_products = None):
    if image_results:
        return [result.entity.get('p_id') for result in image_results if result.entity.get('p_id')]
    elif product_results:
        return [result.entity.get('product_id') for result in product_results if result.entity.get('product_id')]
    elif filtered_products:
        return [item.get('product_id') for item in filtered_products if item.get('product_id')]
    else:
        return None

def sort_weighted_products(all_products):
    sorted_products = dict(sorted(all_products.items(), key=lambda item: item[1]['final_rank_score'], reverse=True))
    return sorted_products

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

def load_config(config_path = 'App/config.yaml'):
    """Load the configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def format_results(results):
    formatted_results = []
    for product_id, details in results.items():
        formatted_result = {
            "Title": details['title'],
            "Description": details['description'],
            'Features': details['features'],
            "Price": details['price'],
            'Average_Rating': details['average_rating'],
            "Main_Category": details['main_category'],
            "Store": details['store'],
            "Categories": ", ".join(details['categories']),
            "Images": details['image_urls'],
            "Score": details['scores']
        }
        if 'final_rank_score' in details:
            formatted_result['Final Rank Score'] = details['final_rank_score']
        formatted_results.append(formatted_result)
    return formatted_results