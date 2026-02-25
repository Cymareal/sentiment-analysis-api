# Sentiment Analysis API

A machine learning API that predicts whether an Amazon review is positive or negative using NLP and Logistic Regression.

## Project Overview

This project builds an end-to-end sentiment analysis system trained on 160,000+ Amazon Digital Music reviews. The model is served via a Flask REST API that accepts review text and returns a sentiment prediction.

## Tech Stack

- Python
- Scikit-learn (TF-IDF, Logistic Regression)
- NLTK (text preprocessing)
- Flask (REST API)
- Pandas & NumPy

## Project Structure
```
sentiment-analysis-api/
├── data/                   # Dataset files
├── notebooks/
│   ├── 01_exploration.ipynb    # Data exploration & preprocessing
│   └── 02_modeling.ipynb       # Model training & evaluation
├── src/
│   ├── preprocess.py       # Text cleaning functions
│   └── train.py            # Training script
├── model/
│   ├── sentiment_model.pkl # Saved model
│   └── vectorizer.pkl      # Saved TF-IDF vectorizer
├── app.py                  # Flask API
└── requirements.txt        # Dependencies
```

## Model Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Negative | 0.84 | 0.91 | 0.87 |
| Positive | 0.90 | 0.83 | 0.87 |
| **Overall Accuracy** | | | **87%** |

## How to Run

1. Clone the repository
2. Create a virtual environment and activate it
3. Install dependencies
4. Run the Flask API
```bash
git clone https://github.com/yourusername/sentiment-analysis-api
cd sentiment-analysis-api
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## API Usage

Send a POST request to `/predict` with a review text:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "This album is absolutely amazing!"}'
```

Response:
```json
{
  "review": "This album is absolutely amazing!",
  "sentiment": "positive"
}
```
## Live Demo

API is live and accessible at:
`https://sentiment-analysis-api-fh04.onrender.com/predict`

Test it with a POST request:
```powershell
(Invoke-WebRequest -Uri https://sentiment-analysis-api-fh04.onrender.com/predict -Method POST -ContentType "application/json" -Body '{"review": "This album is absolutely amazing!"}').Content
```

Expected response:
```json
{
  "review": "This album is absolutely amazing!",
  "sentiment": "positive"
}
```

## Key Learnings

- Handled severe class imbalance (97% positive vs 3% negative) using undersampling
- Learned that high accuracy can be misleading with imbalanced datasets
- Built a complete ML pipeline from raw data to deployed API