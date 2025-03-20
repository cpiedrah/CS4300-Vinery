import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd

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
    # episodes_df = pd.DataFrame(data['episodes'])
    # reviews_df = pd.DataFrame(data['reviews'])
    wines_df = pd.DataFrame(data)

app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
def json_search(query):
    # matches = []
    # merged_df = pd.merge(episodes_df, reviews_df, left_on='id', right_on='id', how='inner')
    # matches = merged_df[merged_df['title'].str.lower().str.contains(query.lower())]
    # matches_filtered = matches[['title', 'descr', 'imdb_rating']]
    # matches_filtered_json = matches_filtered.to_json(orient='records')
    # return matches_filtered_json
    if query:
        # Search is case-insensitive and skips any NaN values
        query_lower = query.lower()
        matches = wines_df[
            wines_df['Wine Name'].str.lower().str.contains(query_lower, na=False) |
            wines_df['Review'].str.lower().str.contains(query_lower, na=False) |
            wines_df['Veriety'].fillna('').str.lower().str.contains(query_lower, na=False)
        ]
    else:
        matches = wines_df
    matches_filtered = matches[['Wine Name', 'Review', 'Rating', 'Veriety']]
    return matches_filtered.to_json(orient='records')

@app.route("/")
def home():
    return render_template('base.html',title="Wine Reviews Search")

@app.route("/search")
def search():
    # text = request.args.get("title")
    # return json_search(text)
    query = request.args.get("query", "")
    return json_search(query)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)