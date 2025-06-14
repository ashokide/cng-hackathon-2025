import logging
import joblib
import re
import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the trained model and scaler
model_path = 'hours_prediction_model.pkl'
scaler_path = 'hours_scaler.pkl'
model = None
scaler = None

if not os.path.exists(model_path):
    logger.error(f"Model file {model_path} not found. Please run train_model.py.")
    exit(1)
if not os.path.exists(scaler_path):
    logger.error(f"Scaler file {scaler_path} not found. Please run train_model.py.")
    exit(1)

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    exit(1)

# Define Pydantic model for request validation
class PredictionInput(BaseModel):
    Number: str = ""
    Short_description: str = ""
    Description: str = ""
    Acceptance_Criteria: str = ""

# Define text cleaning function (aligned with training script)
def clean_text(text: str) -> str:
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower().strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Define prediction endpoint
@app.post("/predict")
async def predict(data: PredictionInput):
    try:
        # Combine and clean text
        combined_text = f"{data.Short_description} {data.Description} {data.Acceptance_Criteria}"
        combined_text = clean_text(combined_text)
        
        # Create DataFrame with all features expected by the model
        input_df = pd.DataFrame({
            'combined_text': [combined_text],
            'text_length': [len(combined_text)],
            'api_count': [combined_text.count('api')],
            'bug_count': [len(re.findall(r'bug|error|fix', combined_text))],
            'feature_count': [len(re.findall(r'feature|add|new', combined_text))],
            'urgent_count': [len(re.findall(r'urgent|critical', combined_text))]
        })
        
        # Predict hours (in scaled space)
        predicted_hours_scaled = model.predict(input_df)[0]
        # Inverse transform to original scale
        predicted_hours = scaler.inverse_transform([[predicted_hours_scaled]])[0][0]
        # Inverse log transformation
        predicted_hours = np.expm1(predicted_hours)
        
        # Log prediction
        logger.info(f"Predicting for Number: {data.Number}, Input Text: {combined_text[:100]}..., Predicted Hours: {predicted_hours:.2f}")
        
        # Return response
        return {
            'Number': data.Number,
            'Predicted Hours': round(predicted_hours)
        }
    except Exception as e:
        logger.error(f"Prediction error for Number {data.Number}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# Health check endpoint
@app.get("/")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

# Run the application (for development; in production, use Uvicorn directly)
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000)