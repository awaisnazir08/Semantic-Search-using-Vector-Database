from pymilvus import connections

def connect_milvus(host, port):
    client = connections.connect("default", host=host, port=port)
    print("Is Milvus connected:", connections.has_connection("default"))
    return client