from flask import Flask, render_template, request
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv('mvs.csv')
df = df.head(500)

# Preprocess
df['fullplot'] = df['fullplot'].fillna('')
df['fullplot'] = df['fullplot'].apply(lambda x: x.lower())
df['fullplot'] = df['fullplot'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# TF-IDF and Cosine Similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['fullplot'])

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        movie = request.form['movie']
        print("User input:", movie)

        # Debug: Print all available movie titles
        print("Available movies:", df['title'].tolist())

        # Match with partial title (case-insensitive)
        matching = df[df['title'].str.lower().str.contains(movie.lower())]
        
        if not matching.empty:
            index = matching.index[0]  # take the first match
            print("Matched movie:", df.loc[index, 'title'])
            
            # Get cosine similarity for the matched movie
            sim_scores = cosine_similarity(tfidf_matrix[index], tfidf_matrix).flatten()
            indices = sim_scores.argsort()[-6:-1][::-1]  # Top 5 recommendations

            recommended_movies = df.iloc[indices]['title'].tolist()
            print("Recommendations:", recommended_movies)

            return render_template('recommendations.html', movie=df.loc[index, 'title'], recommendations=recommended_movies)
        else:
            print("Movie not found in dataset.")
            return render_template('recommendations.html', movie=movie, recommendations=[])

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


