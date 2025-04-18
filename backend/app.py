import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
import pandas as pd
import numpy as np

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'cellartracker_wine_reviews_with_rating_40pages.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    wines_df = pd.DataFrame(data)
    documents = [(x['Wine Name'], x['Variety'], x['Review'], x['Rating'])
                 for x in data if x['Review'] is not None and len(x['Review']) > 20]
    vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .7, min_df = 1)
    td_matrix = vectorizer.fit_transform([x[2] for x in documents if x[2] is not None])
    u, s, v_trans = svds(td_matrix, k=100)

    docs_compressed, s, words_compressed = svds(td_matrix, k=40)
    words_compressed = words_compressed.transpose()

    word_to_index = vectorizer.vocabulary_
    index_to_word = {i:t for t,i in word_to_index.items()}

    words_compressed_normed = normalize(words_compressed, axis = 1)

    td_matrix_np = td_matrix.transpose().toarray()
    td_matrix_np = normalize(td_matrix_np)

    docs_compressed_normed = normalize(docs_compressed)

    # Get the unique varieties from the DataFrame without None values
    unique_varieties = wines_df['Variety'].dropna().unique()

app = Flask(__name__)
CORS(app)

# cosine similarity
def closest_words(word_in, words_representation_in, k = 10):
    if word_in not in word_to_index: return None
    sims = words_representation_in.dot(words_representation_in[word_to_index[word_in],:])
    asort = np.argsort(-sims)[:k+1]
    return [(index_to_word[i],sims[i]) for i in asort[1:]]

def closest_wines(wine_index_in, wine_repr_in, k = 5):
    sims = wine_repr_in.dot(wine_repr_in[wine_index_in,:])
    asort = np.argsort(-sims)[:k+1]
    return [(documents[i][0], documents[i][1], documents[i][2], documents[i][3], sims[i]) for i in asort[1:]]

def closest_wines_to_query(query_vec_in, k = 5, variety_filter=None, year_filter=None):
    sims = docs_compressed_normed.dot(query_vec_in)
    asort = np.argsort(-sims)
    
    # if both variety and year filter do not exist
    if variety_filter is None and year_filter is None:
        return [(documents[i][0], documents[i][1], documents[i][2], documents[i][3], sims[i]) for i in asort[1:]]
    
    added = 0
    res = []
    # if both variety and year filter exist
    if variety_filter is not None and year_filter is not None:
        for i in asort[1:]:
            if documents[i][1] is not None and variety_filter.lower() in documents[i][1].lower() and str(year_filter) in documents[i][0]:
                res.append((documents[i][0], documents[i][1], documents[i][2], documents[i][3], sims[i]))
                added += 1
                if added == k:
                    break
    # if only variety filter exists
    if variety_filter is not None and year_filter is None:
        for i in asort[1:]:
            if documents[i][1] is not None and variety_filter.lower() in documents[i][1].lower():
                res.append((documents[i][0], documents[i][1], documents[i][2], documents[i][3], sims[i]))
                added += 1
                if added == k:
                    break
    # if only year filter exists
    if variety_filter is None and year_filter is not None:
        for i in asort[1:]:
            if documents[i][1] is not None and str(year_filter) in documents[i][0]:
                res.append((documents[i][0], documents[i][1], documents[i][2], documents[i][3], sims[i]))
                added += 1
                if added == k:
                    break
    return res

# Sample search using json with pandas
def json_search(query):
    variety_filter = None
    year_filter = None
    if query:
        # Search is case-insensitive
        query_lower = query.lower()
        # If the query contains any of the varieties in unique varieties
        for variety in unique_varieties:
            # Need to avoid matching substrings like "rice" in "licorice"
            if f" {variety.lower()} " in query_lower:
                variety_filter = variety
                query_lower = query_lower.replace(variety.lower(), '')
                print(f"Variety filter: {variety_filter}")
        # If the query contains a year (4 digits)
        for word in query_lower.split():
            if len(word) == 4 and word.isdigit():
                query_lower = query_lower.replace(word, '')
                year_filter = word
        # Get tf-idf representation of the query
        query_tfidf = vectorizer.transform([query_lower]).toarray()
        # Normalize the query vector
        query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()
        # Find the closest wines to the query vector
        matches = closest_wines_to_query(query_vec, k = 20, variety_filter=variety_filter, year_filter=year_filter)
        # Convert matches to DataFrame for easier manipulation
        matches = pd.DataFrame(matches, columns=['Wine Name', 'Variety', 'Review', 'Rating', 'Similarity'])
    else:
        # If no query, return all wines with no similarity score
        matches = wines_df
        matches['Similarity'] = None
    
    matches_filtered = matches[['Wine Name', 'Review', 'Rating', 'Variety', 'Similarity']]
    return matches_filtered.to_json(orient='records')

@app.route("/")
def home():
    return render_template('base.html',title="Wine Reviews Search")

@app.route("/search")
def search():
    query = request.args.get("query", "")
    return json_search(query)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)