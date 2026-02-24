from flask import Flask, request, jsonify
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('model/sentiment_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data['review']
    cleaned = clean_text(review)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    sentiment = 'positive' if prediction == 1 else 'negative'
    return jsonify({'sentiment': sentiment, 'review': review})

if __name__ == '__main__':
    app.run(debug=True)