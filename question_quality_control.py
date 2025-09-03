def sim_search(response, k, vector_store):
    for i in range(len(response)):
        docs = vector_store.similarity_search_with_score(response[str(i+1)]["Question"], k=k)
        for doc in docs:
            score = doc[1]
            content = doc[0].page_content
            if score < 0.6:
                del response[str(i+1)]
                continue
    return response