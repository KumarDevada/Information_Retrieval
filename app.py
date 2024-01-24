from flask import Flask, render_template, request, send_file
import os
import gensim
from gensim import corpora, similarities
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import pickle
from datetime import datetime

app = Flask(__name__)


def preprocess_text(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS:
            result.append(token)
    return result

def load_index():
    try:
        with open('./index/dictionary.pkl', 'rb') as dictionary_file:
            dictionary = pickle.load(dictionary_file)

        with open('./index/inverted_index.pkl', 'rb') as index_file:
            index = pickle.load(index_file)

        return dictionary, index
    except Exception as e:
        print(f"An error occurred while loading index files: {e}")
        return None, None

def process_query(query, dictionary, index):
    if dictionary is None or index is None:
        print("Index files not loaded. Exiting.")
        return []

    processed_query = preprocess_text(query)
    query_bow = dictionary.doc2bow(processed_query)

    index = similarities.MatrixSimilarity.load('./index/similarity_index.index')
    sims = index[query_bow]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    relevant_docs = [(doc_id, score) for doc_id, score in sims if score > 0.0]
    top_10_docs = relevant_docs[:10]

    return top_10_docs


@app.route('/open_doc/<doc_id>', methods=['GET'])
def open_document(doc_id):
    doc_path = f'./data/{doc_id}.txt'  # Update this path to your document directory
    if os.path.exists(doc_path):
        return send_file(doc_path, as_attachment=True)
    else:
        return "Document not found."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    
    query = request.form['query']

    dictionary, _ = load_index()

    if dictionary:
        index = similarities.MatrixSimilarity.load('./index/similarity_index.index')

        start_time = datetime.now()
        results = process_query(query, dictionary, index)

        end_time = datetime.now()
        time_taken = (end_time - start_time).total_seconds()
        if results:
            response = "<h2>Top 10 Relevant Documents</h2>"
            for doc_id, score in results:
                response += f"<a href='/open_doc/{doc_id}'><p>Document ID: {doc_id}, Similarity Score: {score:.4f}</p><a>"
            response += f"<p id='benchmark' >Benchmark time taken: {time_taken} seconds"
            return response
        else:
            return f"<p>No relevant documents found.</p><p id='benchmark' >Benchmark time taken : {time_taken} seconds</p>"
    else:
        return "<p>Dictionary not loaded. Exiting.</p>"

if __name__ == "__main__":
    app.run(debug=True)
