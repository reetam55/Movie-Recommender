from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load dataset
movies = pd.read_csv('mvs.csv')

# Fill NaN in 'plot' column
movies['plot'] = movies['plot'].fillna('')

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['plot'])

# Fit NearestNeighbors model
model = NearestNeighbors(n_neighbors=6, metric='cosine')
model.fit(tfidf_matrix)

movies = movies.reset_index()

def get_recommendations(title):
    title = title.lower()
    matches = movies[movies['title'].str.lower() == title]
    if matches.empty:
        return None  # <-- return None if movie not found
    idx = matches.index[0]
    distances, indices = model.kneighbors(tfidf_matrix[idx], n_neighbors=6)
    recommended_indices = indices.flatten()[1:]  # Skip the input movie itself
    return movies['title'].iloc[recommended_indices].tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie = request.form['movie']
    recommendations = get_recommendations(movie)
    
    if recommendations is None:
        return render_template('index.html', movie=movie, recommendations=[], not_found=True)
    else:
        return render_template('index.html', movie=movie, recommendations=recommendations, not_found=False)


if __name__ == '__main__':
    app.run(debug=True)
