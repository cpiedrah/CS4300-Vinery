import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
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
# json_file_path = os.path.join(current_directory, 'cellartracker_wine_reviews_with_rating_40pages.json')
json_file_path = os.path.join(current_directory, 'cellartracker_wine_reviews.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    wines_df = pd.DataFrame(data)
    documents = [(x['Wine Name'], x['Variety'], x['Review'], x['Rating'], x['Location'])
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


    ratings = np.array([x[3].split(' ')[-2] for x in documents])

    # if the rating is not a number, set it to 0
    ratings = [float(r) if not ':' in r else 0 for r in ratings]
    ratings = np.array(ratings)
    ratings = ratings / 100

    # load location map from location_dict_deduped.json
    location_map = {}
    with open(os.path.join(current_directory, 'location_dict_deduped.json'), 'r') as file:
        location_map = json.load(file)
    


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

def closest_wines_to_query(query_vec_in, k = 5, variety_filter=None, year_filter=None, color_filter=None, location_filter=None):
    sims = docs_compressed_normed.dot(query_vec_in)
    sims = (sims + ratings) / 2
    asort = np.argsort(-sims)
    
    ROSE_KEYWORDS = ['rosé', 'rose']
    RED_KEYWORDS = [
        'cabernet', 'merlot', 'pinot noir', 'syrah', 'zinfandel', 'tempranillo',
        'grenache', 'nebbiolo', 'sangiovese', 'malbec', 'bordeaux', 'red'
    ]
    WHITE_KEYWORDS = [
        'chardonnay', 'sauvignon', 'riesling', 'pinot gris', 'white', 'viognier',
        'marsanne', 'roussanne', 'gewürztraminer', 'semillon', 'chenin blanc'
    ]

    def classify_color(variety, title, designation):
        def field_to_str(field):
            return field[0].lower() if isinstance(field, list) and field else field.lower() if isinstance(field, str) else ''
    
        v = field_to_str(variety)
        t = field_to_str(title)
        d = field_to_str(designation)

        if any(keyword in v or keyword in t or keyword in d for keyword in ROSE_KEYWORDS):
            return 'rosé'
        if any(keyword in v or keyword in t or keyword in d for keyword in WHITE_KEYWORDS):
            return 'white'
        if any(keyword in v or keyword in t or keyword in d for keyword in RED_KEYWORDS):
            return 'red'
        return 'unknown'
    
    added = 0
    res = []

    for i in asort[1:]:
        # Full metadata row
        wine_data = data[i]

        title, variety, review, rating, location = documents[i][:5]
        designation = wine_data.get("designation", [""])[0] if isinstance(wine_data.get("designation"), list) else ""
        # FILTERS
        if variety_filter and (not variety or variety_filter.lower() not in variety.lower()):
            continue
        if year_filter and str(year_filter) not in title:
            continue
        if color_filter and classify_color(variety, title, designation) != color_filter:
            continue
        if location_filter and (not location or (location.lower() not in location_filter.lower() and all(location.lower() not in loc.lower() for loc in location_map.get(location_filter, [])))):
            continue
        res.append((title, variety, review, rating, location, sims[i]))
        added += 1
        if added == k:
            break

    return res

# Sample search using json with pandas
def json_search(query):
    variety_filter = None
    year_filter = None
    color_filter = None
    location_filter = None

    if query:
        # Search is case-insensitive
        query_lower = query.lower()

        if 'red wine' in query_lower:
            color_filter = 'red'
            query_lower = query_lower.replace('red wine', '')
        elif 'white wine' in query_lower:
            color_filter = 'white'
            query_lower = query_lower.replace('white wine', '')
        elif 'rosé' in query_lower or 'rose' in query_lower:
            color_filter = 'rosé'
            query_lower = query_lower.replace('rosé', '').replace('rose', '')

        VARIETY_COLOR_MAP = {
        'chardonnay': 'white',
        'sauvignon blanc': 'white',
        'riesling': 'white',
        'pinot gris': 'white',
        'viognier': 'white',
        'marsanne': 'white',
        'roussanne': 'white',
        'gewürztraminer': 'white',
        'semillon': 'white',
        'chenin blanc': 'white',
        'cabernet sauvignon': 'red',
        'merlot': 'red',
        'pinot noir': 'red',
        'syrah': 'red',
        'zinfandel': 'red',
        'tempranillo': 'red',
        'grenache': 'red',
        'nebbiolo': 'red',
        'sangiovese': 'red',
        'malbec': 'red'
    }
        # infer color of wine from wine name
        for variety_name, color in VARIETY_COLOR_MAP.items():
            if variety_name in query_lower:
                color_filter = color
                break
        # If the query contains any of the varieties in unique varieties
        for variety in unique_varieties:
            # Need to add spaces to avoid matching 'licorice' with 'rice' as a varietal
            if f" {variety.lower()} " in f" {query_lower} ":
                variety_filter = variety
                query_lower = query_lower.replace(variety.lower(), '')
                break  # Stop at first match
        # If the query contains a year (4 digits)
        for word in query_lower.split():
            if len(word) == 4 and word.isdigit():
                query_lower = query_lower.replace(word, '')
                year_filter = word
        for location in location_map.keys():
            if location.lower() in query_lower:
                query_lower = query_lower.replace(location.lower(), '')
                location_filter = location

        # Get tf-idf representation of the query
        query_tfidf = vectorizer.transform([query_lower]).toarray()
        # Normalize the query vector
        query_vec = normalize(np.dot(query_tfidf, words_compressed)).squeeze()
        # Find the closest wines to the query vector
        matches = closest_wines_to_query(
        query_vec,
        k=20,
        variety_filter=variety_filter,
        year_filter=year_filter,
        color_filter=color_filter,
        location_filter=location_filter
        )
    else:
        matches = wines_df.copy()
        matches['Similarity'] = None

        columns_to_use = ['Wine Name', 'Review', 'Rating', 'Location', 'Variety']
        columns_present = [col for col in columns_to_use if col in matches.columns]
        matches = matches[columns_present]
    
    matches = pd.DataFrame(matches, columns=['Wine Name', 'Variety', 'Review', 'Rating', 'Location', 'Similarity'])
    matches_filtered = matches[['Wine Name', 'Review', 'Rating', 'Variety', 'Location', 'Similarity']]

    
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
