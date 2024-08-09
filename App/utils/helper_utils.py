from PIL import Image

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