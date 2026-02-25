from flask import Flask, request, jsonify
import joblib
from src.preprocess import clean_text

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('model/sentiment_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

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