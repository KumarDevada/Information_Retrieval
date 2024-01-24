import os
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import pickle
import time

def preprocess_text(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS:
            result.append(token)
    return result

def create_index():
    doc_dir = "../data/"
    file_list = [
        file for file in os.listdir(doc_dir) if file.endswith('.txt')
        and os.path.isfile(os.path.join(doc_dir, file))
    ]

    print(f"Total files to process: {len(file_list)}")
    start_time = time.time()

    documents = []
    for idx, file_name in enumerate(file_list, start=1):
        with open(os.path.join(doc_dir, file_name), 'r', encoding='utf-8') as file:
            text = file.read()
            processed_text = preprocess_text(text)
            documents.append(processed_text)
        
        # Log progress for every 1000 files processed
        if idx % 1000 == 0 or idx == len(file_list):
            print(f"Processed {idx}/{len(file_list)} files")

        # Break after processing 5000 files
        # if idx == 5000:
        #     break

    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    tfidf = models.TfidfModel(corpus)
    index = tfidf[corpus]

    # Saving the index files
    index_folder = '../index/'
    
    try:
        os.makedirs(index_folder, exist_ok=True)
        
        with open(os.path.join(index_folder, 'dictionary.pkl'), 'wb') as dictionary_file:
            pickle.dump(dictionary, dictionary_file)

        with open(os.path.join(index_folder, 'inverted_index.pkl'), 'wb') as index_file:
            pickle.dump(index, index_file)
        
        print("Index files saved successfully.")
    except Exception as e:
        print(f"An error occurred while saving index files: {e}")

    end_time = time.time()
    print(f"Indexing completed in {(end_time - start_time):.2f} seconds")

    # Create and save the similarity index
    similarity_index_path = os.path.join(index_folder, 'similarity_index.index')
    try:
        similarity_index = gensim.similarities.MatrixSimilarity(tfidf[corpus])
        similarity_index.save(similarity_index_path)
        print("Similarity index saved successfully.")
    except Exception as e:
        print(f"An error occurred while saving the similarity index: {e}")

if __name__ == "__main__":
    create_index()
