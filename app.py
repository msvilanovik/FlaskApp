from config import OPENAI_API_KEY
from gensim.models import KeyedVectors
from flask import Flask, request, render_template
import openai
from scipy.spatial.distance import cosine

app = Flask(__name__)

openai.api_key = OPENAI_API_KEY


@app.route('/static/<path:filename>')
def serve_static(filename):
  return app.send_static_file(filename)


@app.route('/')
def search_form():
  return render_template('search_form.html')


@app.route('/search')
def search():

    query = request.args.get('query')

    bin_file_path = 'foaf_word2vec (4).bin'

    foaf_word_vectors = KeyedVectors.load(bin_file_path)

    if query in foaf_word_vectors.wv:

        search_term_vector = foaf_word_vectors.wv.get_vector(query)

        similarities = {}
        for entity in foaf_word_vectors.wv.index_to_key:
            similarity = 1 - cosine(search_term_vector, foaf_word_vectors.wv.get_vector(entity))
            similarities[entity] = similarity

        similar_entities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:10]

        results = [entity[0] for entity in similar_entities]
    else:
        results = []

    return render_template('search_results.html', query=query, results=results)

#The following code is for comining foaf and schema vocabularies together so we can compare the search term with both
# @app.route('/search')
# def search():
#
#     query = request.args.get('query')
#
#     bin_file_path = 'foaf_word2vec (4).bin'
#     bin_file_path2 = 'schema_word2vec.bin'
#
#     foaf_word_vectors = KeyedVectors.load(bin_file_path)
#     schema_word_vectors = KeyedVectors.load(bin_file_path2)
#
#     combined_word_vectors = foaf_word_vectors
#     combined_word_vectors.add(schema_word_vectors)
#
#     if query in combined_word_vectors.wv:
#         search_term_vector = combined_word_vectors.wv.get_vector(query)
#
#         similarities = {}
#         for entity in combined_word_vectors.wv.index_to_key:
#             similarity = 1 - cosine(search_term_vector, combined_word_vectors.wv.get_vector(entity))
#             similarities[entity] = similarity
#
#         similar_entities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:10]
#
#         results = [entity[0] for entity in similar_entities]
#     else:
#         results = []
#
#     return render_template('search_results.html', query=query, results=results)
# ------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    app.run()
