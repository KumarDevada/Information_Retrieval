import os
import gensim
from gensim import corpora, similarities
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import pickle

def preprocess_text(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS:
            result.append(token)
    return result

def load_index():
    try:
        with open('../index/dictionary.pkl', 'rb') as dictionary_file:
            dictionary = pickle.load(dictionary_file)

        with open('../index/inverted_index.pkl', 'rb') as index_file:
            index = pickle.load(index_file)

        return dictionary, index
    except Exception as e:
        print(f"An error occurred while loading index files: {e}")
        return None, None

def process_query(query, dictionary, index):
    if dictionary is None or index is None:
        print("Index files not loaded. Exiting.")
        return

    processed_query = preprocess_text(query)
    query_bow = dictionary.doc2bow(processed_query)

    index = similarities.MatrixSimilarity.load('../index/similarity_index.index')
    sims = index[query_bow]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    relevant_docs = [(doc_id, score) for doc_id, score in sims if score > 0.0]
    top_10_docs = relevant_docs[:10]

    return top_10_docs

if __name__ == "__main__":
    dictionary, _ = load_index()

    if dictionary:
        index = similarities.MatrixSimilarity.load('../index/similarity_index.index')
        
        while True:
            query = input("Enter your query (type 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            
            results = process_query(query, dictionary, index)
            if results:
                print("\nTop relevant documents:")
                for doc_id, score in results:
                    print(f"Document ID: {doc_id}, Similarity Score: {score:.4f}")
            else:
                print("No relevant documents found.")
    else:
        print("Dictionary not loaded. Exiting.")
