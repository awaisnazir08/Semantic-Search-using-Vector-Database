from .milvus_db.setup import MilvusManager
from .model.get_model import Model
from .interface.ui import Interface
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

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    milvus = MilvusManager()
    embeddings_generation_model = Model()
    
    milvus.connect()
    products_collection, images_collection, search_params = milvus.setup_collections()
    
    model, preprocess, tokenizer = embeddings_generation_model.load_model()
    
    interface = Interface(products_collection, images_collection, search_params, model, preprocess, tokenizer, device)
    
    interface.launch_interface()

if __name__ == "__main__":
    main()

