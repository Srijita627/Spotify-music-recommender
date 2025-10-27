from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv('songs.csv')
df['lyrics_summary'] = df['lyrics_summary'].fillna('')

# Create TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['lyrics_summary'])
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    input_song = ""

    if request.method == 'POST':
        input_song = request.form.get('song')

        if input_song in df['title'].values:
            idx = df[df['title'] == input_song].index[0]
            scores = list(enumerate(similarity_matrix[idx]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            scores = scores[1:6]
            recommendations = [df.iloc[i[0]]['title'] for i in scores]
        else:
            recommendations = ["Song not found in dataset"]

    return render_template('index.html', input_song=input_song, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)

