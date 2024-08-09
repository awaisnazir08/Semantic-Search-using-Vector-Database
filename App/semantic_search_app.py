from .milvus_db.connection import connect_milvus
from .milvus_db.setup import setup_collections
from .interface.ui import launch_interface
import torch
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    db,
    MilvusClient
)

# Constants
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    host = "192.168.1.103"
    port = 19530
    
    # Connect to Milvus
    client = connect_milvus(host, port)
    
    # Setup collections
    products_collection, images_collection, search_params = setup_collections(client)
    
    # Launch interface
    launch_interface(products_collection, images_collection, search_params, device)

if __name__ == "__main__":
    main()

