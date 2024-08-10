import yaml
from pymilvus import utility, db, Collection, connections
from ..utils.helper_utils import load_config
class MilvusManager:
    def __init__(self):
        self.config = load_config()
        self.host = self.config['milvus']['host']
        self.port = self.config['milvus']['port']
        self.client = None
        self.search_params = self.config['search_params']
        self.index_params = self.config['index_params']

    def connect(self):
        """Establish a connection to the Milvus server."""
        self.client = connections.connect("default", host=self.host, port=self.port)
        print("Is Milvus connected:", connections.has_connection("default"))

    def disconnect(self):
        """Close the connection to the Milvus server."""
        connections.disconnect("default")
        print("Milvus connection closed.")
    
    def _use_database(self, database_name=None):
        """Switch to the specified Milvus database."""
        db_name = database_name or self.config['database']['database_name']
        db.using_database(db_name)
    
    def list_collections(self):
        """List all collections in the current database."""
        collections = utility.list_collections()
        print("Collections:", collections)
        return collections
    
    def _get_collection(self, collection_name):
        """Retrieve a collection by name."""
        return Collection(collection_name)
    
    def _create_index(self, collection, field_name):
        """Create an index on a specified field in the collection."""
        collection.create_index(field_name=field_name, index_params=self.config['index_params'])
    
    def _load_collection(self, collection):
        """Load a collection into memory."""
        collection.load()
    
    def setup_collections(self):
        """Full setup process for collections, including indexing and loading."""
        self._use_database()
        self.list_collections()
        
        self.products_collection = self._get_collection(self.config['database']['products_collection'])
        self._create_index(self.products_collection, "title_vector")
        self._load_collection(self.products_collection)
        
        self.images_collection = self._get_collection(self.config['database']['images_collection'])
        self._create_index(self.images_collection, "image_vector")
        self._load_collection(self.images_collection)
        
        return self.products_collection, self.images_collection, self.search_params
    
    def drop_index(self, collection):
        collection.drop_index()
    
    def release_collection(self, collection):
        collection.release()

if __name__ == '__main__':
    # Usage
    milvus_manager = MilvusManager()
    milvus_manager.connect()
    milvus_manager.setup_collections()
    milvus_manager.disconnect()