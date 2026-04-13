import chromadb

db_path = r"C:\Users\tywal\Desktop\bioinsight-radar\bioinsight_db"
client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name="bioinsight_library")
second_entry = collection.get(limit=1, offset=3, include=["embeddings", "documents", "metadatas"])
print(second_entry)