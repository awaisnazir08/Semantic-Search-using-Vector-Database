from pymilvus import utility
from pymilvus import connections, MilvusClient, db, Collection
host = "141.195.16.189"
port = 40238 # Mapping for 19530 (default Milvus port)

# Connect to Milvus
client = connections.connect("default", host=host, port=port)

# Check if the connection is established
print("Is Milvus connected:", connections.has_connection("default"))
db.using_database('Products')
# Optional: List collections to confirm the connection
print("Collections:", utility.list_collections())

collection_name = "products"
collection = Collection(collection_name)

# Get the number of entities
num_entities = collection.num_entities
print(f"Number of items in collection '{collection_name}': {num_entities}")

