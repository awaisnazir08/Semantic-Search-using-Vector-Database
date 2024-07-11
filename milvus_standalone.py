from pymilvus import connections, utility, MilvusException, db

connections.connect(host="localhost", port="19530")

try:
    # List all collections
    collections = utility.list_collections()
    databases = db.list_database()
    print(f'List all Databases: \n', databases)
    print(f"List all collections:\n", collections)
except MilvusException as e:
    print(e)