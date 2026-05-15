#!/bin/bash

echo "========================================"
echo "  AI Resume Analyzer — Setup Script"
echo "========================================"

echo ""
echo "[1/3] Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "[2/3] Downloading spaCy English model..."
python3 -m spacy download en_core_web_sm

echo ""
echo "[3/3] Downloading NLTK data..."
python3 -c "
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
print('NLTK data downloaded successfully.')
"

echo ""
echo "========================================"
echo "  Setup complete! Run the app with:"
echo "  streamlit run app.py"
echo "========================================"
