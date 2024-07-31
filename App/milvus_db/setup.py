from pymilvus import utility, db, Collection

def setup_collections(client):
    print("Collections:", utility.list_collections())
    
    db.using_database("Products")
    
    products_collection = Collection('products')
    images_collection = Collection('images')
    
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 256}
    }
    
    search_params = {
        "metric_type": "COSINE", 
        "offset": 0, 
        "ignore_growing": False, 
        "params": {"nprobe": 20}
    }
    
    # index_params = {
    #     "index_type": "HNSW",
    #     "metric_type": "COSINE",  # or "IP" for Inner Product
    #     "params": {"M": 64, "efConstruction": 300}
    # }
    
    # products_collection.release()
    # products_collection.drop_index()
    
    # images_collection.release()
    # images_collection.drop_index()
    
    products_collection.create_index(field_name="title_vector", index_params=index_params)
    products_collection.load()
    
    images_collection.create_index(field_name="image_vector", index_params=index_params)
    images_collection.load()
    
    return products_collection, images_collection, search_params
