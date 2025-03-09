from retrieval import StoreEnum


def retrieval_process(store_client, chunks, query_embedding_data, store_type, collection_name, top_k = 3):
    if store_type == StoreEnum.FAISS:
        distances, indices = store_client.search(query_embedding_data, top_k)
        results = []
        for i in range(top_k):
            result_chunk = chunks[indices[0][i]]
            result_distance = distances[0][i]
            print(f"result_chunk: {result_chunk}, result_distance: {result_distance}")
            results.append(result_chunk)
        
        return results
    elif store_type == StoreEnum.CHROMA:
        db_results = store_client.search(query_embedding_data, collection_name, top_k)
        results = []
        for doc_id, doc, score in zip(db_results['ids'][0], db_results['documents'][0], db_results['distances'][0]):
            print(f"doc_id: {doc_id}, doc: {doc}, score: {score}")
            results.append(doc)
            
        return results
    else:
        raise ValueError(f"Invalid store type: {store_client}")
    
    