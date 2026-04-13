# ================================================
# PICKAFLIX — Content-Based Filtering API
# Uses TF-IDF + Cosine Similarity (as per PPT)
# ================================================
# Install dependencies:
#   pip install flask flask-cors requests scikit-learn pandas numpy
#
# Run locally:
#   python app.py
# ================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
CORS(app, origins="*")
TMDB_KEY = '5ce7e04ed7ea720c4e8c8f11b637e9ec'
TMDB_BASE = 'https://api.themoviedb.org/3'
IMG_BASE  = 'https://image.tmdb.org/t/p/w500'

# ── STEP 1: Fetch movie data from TMDB ──────────
def fetch_popular_movies(pages=10):
    """Fetch a large pool of movies to build our similarity matrix from."""
    movies = []
    for page in range(1, pages + 1):
        try:
            res = requests.get(f'{TMDB_BASE}/movie/popular', params={
                'api_key': TMDB_KEY, 'page': page, 'language': 'en-US'
            }, timeout=10)
            data = res.json()
            movies.extend(data.get('results', []))
        except:
            pass
    return movies

def fetch_movie_details(movie_id):
    """Get full details including genres and keywords for a movie."""
    try:
        details = requests.get(f'{TMDB_BASE}/movie/{movie_id}', params={
            'api_key': TMDB_KEY, 'append_to_response': 'keywords,credits'
        }, timeout=10).json()
        return details
    except:
        return {}

def build_feature_string(movie):
    """
    Build a combined feature string for each movie.
    As per PPT: genres + keywords + description + cast + directors
    This string is what gets vectorized.
    """
    parts = []

    # Overview / description (TF-IDF)
    overview = movie.get('overview', '')
    if overview:
        parts.append(overview)

    # Genres (Count Vectorization)
    genres = movie.get('genres', [])
    if genres:
        genre_str = ' '.join([g['name'].replace(' ', '') for g in genres])
        parts.append(genre_str + ' ' + genre_str)  # double weight genres

    # Keywords
    keywords = movie.get('keywords', {}).get('keywords', [])
    if keywords:
        kw_str = ' '.join([k['name'].replace(' ', '') for k in keywords[:10]])
        parts.append(kw_str)

    # Cast (top 5 actors)
    cast = movie.get('credits', {}).get('cast', [])
    if cast:
        cast_str = ' '.join([c['name'].replace(' ', '') for c in cast[:5]])
        parts.append(cast_str)

    # Directors
    crew = movie.get('credits', {}).get('crew', [])
    directors = [c['name'].replace(' ', '') for c in crew if c.get('job') == 'Director']
    if directors:
        parts.append(' '.join(directors))

    return ' '.join(parts).lower()

# ── IN-MEMORY CACHE ──────────────────────────────
movie_pool = []       # list of movie dicts with features
feature_matrix = None # TF-IDF cosine similarity matrix
vectorizer = None

def build_similarity_matrix():
    """
    STEP 2 from PPT: Build the similarity matrix using TF-IDF + Cosine Similarity.
    This runs once on startup.
    """
    global movie_pool, feature_matrix, vectorizer

    print("🎬 Fetching movies from TMDB...")
    raw_movies = fetch_popular_movies(pages=15)  # ~300 movies

    print(f"📦 Building feature vectors for {len(raw_movies)} movies...")
    enriched = []
    for i, m in enumerate(raw_movies[:200]):  # limit to 200 for speed
        details = fetch_movie_details(m['id'])
        if details:
            feature_str = build_feature_string(details)
            if feature_str.strip():
                enriched.append({
                    'id':           m['id'],
                    'title':        m.get('title', ''),
                    'overview':     m.get('overview', ''),
                    'poster_path':  m.get('poster_path', ''),
                    'release_date': m.get('release_date', ''),
                    'vote_average': m.get('vote_average', 0),
                    'genres':       [g['name'] for g in details.get('genres', [])],
                    'features':     feature_str,
                })
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1} movies...")

    movie_pool = enriched

    # STEP 2: TF-IDF Vectorization → Cosine Similarity Matrix
    print("🔢 Building TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2)
    )
    feature_strings = [m['features'] for m in movie_pool]
    tfidf_matrix = vectorizer.fit_transform(feature_strings)

    # Cosine Similarity matrix (n_movies × n_movies)
    print("📐 Computing cosine similarity matrix...")
    feature_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    print(f"✅ Ready! Matrix shape: {feature_matrix.shape}")

# ── API ENDPOINTS ────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'movies_loaded': len(movie_pool)})

@app.route('/recommend', methods=['GET'])
def recommend():
    """
    STEPS 3-5 from PPT:
    1. Find the movie by name in our pool
    2. Get its row from the similarity matrix
    3. Sort by similarity score descending
    4. Return top N results
    """
    query = request.args.get('movie', '').strip().lower()
    n     = int(request.args.get('n', 10))

    if not query:
        return jsonify({'error': 'Please provide a movie name'}), 400

    if not movie_pool or feature_matrix is None:
        return jsonify({'error': 'Recommendation engine not ready yet'}), 503

    # STEP 3: Find the movie in our pool
    # Try exact match first, then partial
    movie_index = None
    for i, m in enumerate(movie_pool):
        if m['title'].lower() == query:
            movie_index = i
            break

    if movie_index is None:
        for i, m in enumerate(movie_pool):
            if query in m['title'].lower():
                movie_index = i
                break

    # If not found in pool, search TMDB and use feature-based query
    if movie_index is None:
        # Search TMDB for the movie
        try:
            search_res = requests.get(f'{TMDB_BASE}/search/movie', params={
                'api_key': TMDB_KEY, 'query': query, 'page': 1
            }, timeout=10).json()
            results = search_res.get('results', [])
            if not results:
                return jsonify({'error': f'Movie "{query}" not found'}), 404

            # Get details for the searched movie
            found = results[0]
            details = fetch_movie_details(found['id'])
            query_features = build_feature_string(details)

            # Vectorize the query movie and compare to our matrix
            query_vec = vectorizer.transform([query_features])
            sim_scores = cosine_similarity(query_vec, vectorizer.transform(
                [m['features'] for m in movie_pool]
            ))[0]

            # Sort by score descending
            sorted_indices = np.argsort(sim_scores)[::-1][:n]
            recommendations = []
            for idx in sorted_indices:
                m = movie_pool[idx]
                if m['id'] != found['id']:  # exclude the input movie itself
                    recommendations.append({
                        'id':           m['id'],
                        'title':        m['title'],
                        'poster_path':  m['poster_path'],
                        'release_year': m['release_date'][:4] if m['release_date'] else 'N/A',
                        'vote_average': round(m['vote_average'], 1),
                        'genres':       m['genres'],
                        'overview':     m['overview'],
                        'similarity':   round(float(sim_scores[idx]), 3),
                    })

            return jsonify({
                'query':           found['title'],
                'recommendations': recommendations[:n]
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # STEP 4: Get similarity scores for this movie
    sim_scores = list(enumerate(feature_matrix[movie_index]))

    # STEP 5: Sort by similarity descending, skip the movie itself (score=1.0)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != movie_index][:n]

    recommendations = []
    for idx, score in sim_scores:
        m = movie_pool[idx]
        recommendations.append({
            'id':           m['id'],
            'title':        m['title'],
            'poster_path':  m['poster_path'],
            'release_year': m['release_date'][:4] if m['release_date'] else 'N/A',
            'vote_average': round(m['vote_average'], 1),
            'genres':       m['genres'],
            'overview':     m['overview'],
            'similarity':   round(float(score), 3),
        })

    return jsonify({
        'query':           movie_pool[movie_index]['title'],
        'recommendations': recommendations
    })

@app.route('/search', methods=['GET'])
def search():
    """Search movies from TMDB directly (for autocomplete)."""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify([])
    try:
        res = requests.get(f'{TMDB_BASE}/search/movie', params={
            'api_key': TMDB_KEY, 'query': query, 'page': 1
        }, timeout=10).json()
        results = res.get('results', [])[:8]
        return jsonify([{
            'id':    m['id'],
            'title': m['title'],
            'year':  m.get('release_date', '')[:4],
            'poster': IMG_BASE + m['poster_path'] if m.get('poster_path') else None,
        } for m in results if m.get('title')])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── STARTUP ──────────────────────────────────────
# Build matrix on startup (required for gunicorn)
print("🚀 Starting PickAflix CBT Recommendation Engine...")
build_similarity_matrix()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
