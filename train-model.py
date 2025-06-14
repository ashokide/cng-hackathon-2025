import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import re
import logging
from tqdm import tqdm
import os
import xgboost as xgb
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')

# Step 1: Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_improved.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Step 2: Load and preprocess the dataset
logger.info("Loading rm_story.csv")
try:
    chunks = pd.read_csv('rm_story.csv', encoding='utf-8', chunksize=10000)
    df = pd.concat([chunk for chunk in tqdm(chunks, desc="Loading CSV")])
except UnicodeDecodeError:
    logger.warning("UTF-8 encoding failed, trying latin1")
    chunks = pd.read_csv('rm_story.csv', encoding='latin1', chunksize=10000)
    df = pd.concat([chunk for chunk in tqdm(chunks, desc="Loading CSV")])
except FileNotFoundError:
    logger.error("rm_story.csv not found. Please ensure the file is in the same directory.")
    exit(1)

logger.info(f"Loaded {len(df)} records")

# Log data quality checks
logger.info("Checking for missing values")
missing_counts = df[['Short description', 'Description', 'Acceptance Criteria (Text Version)', 'Estimated Hours']].isna().sum()
logger.info(f"Missing values: {missing_counts.to_dict()}")

# Drop rows where all text fields are empty or NaN
logger.info("Dropping rows with all empty text fields")
text_columns = ['Short description', 'Description', 'Acceptance Criteria (Text Version)']
df['all_text_empty'] = df[text_columns].apply(lambda x: (x.isna() | (x == '')).all(), axis=1)
df = df[~df['all_text_empty']].drop(columns=['all_text_empty'])
logger.info(f"Records after dropping empty text rows: {len(df)}")

# Cap outliers in Estimated Hours
logger.info("Capping outliers in Estimated Hours")
hours_cap = df['Estimated Hours'].quantile(0.99)
df['Estimated Hours'] = df['Estimated Hours'].clip(upper=hours_cap)
logger.info(f"Capped Estimated Hours at 99th percentile: {hours_cap:.2f}")

# Log Estimated Hours statistics
logger.info("Analyzing Estimated Hours distribution")
logger.info(f"Mean: {df['Estimated Hours'].mean():.2f}, Median: {df['Estimated Hours'].median():.2f}, "
            f"Std: {df['Estimated Hours'].std():.2f}, Min: {df['Estimated Hours'].min()}, "
            f"Max: {df['Estimated Hours'].max()}")

# Apply log transformation
logger.info("Applying log transformation to Estimated Hours")
df['Estimated Hours'] = np.log1p(df['Estimated Hours'].clip(lower=0))
logger.info(f"Post-log transform - Mean: {df['Estimated Hours'].mean():.2f}, "
            f"Median: {df['Estimated Hours'].median():.2f}")

# Handle missing values
logger.info("Handling missing values")
df['Short description'] = df['Short description'].fillna('')
df['Description'] = df['Description'].fillna('')
df['Acceptance Criteria (Text Version)'] = df['Acceptance Criteria (Text Version)'].fillna('')
df['Estimated Hours'] = df['Estimated Hours'].fillna(df['Estimated Hours'].median())
logger.info("Missing values handled")

# Combine text columns and add features
logger.info("Combining text columns and extracting features")
df['combined_text'] = (df['Short description'].astype(str) + ' ' +
                      df['Description'].astype(str) + ' ' +
                      df['Acceptance Criteria (Text Version)'].astype(str))
df['text_length'] = df['combined_text'].str.len()
df['api_count'] = df['combined_text'].str.count('api')
df['bug_count'] = df['combined_text'].str.count('bug|error|fix')
df['feature_count'] = df['combined_text'].str.count('feature|add|new')
df['urgent_count'] = df['combined_text'].str.count('urgent|critical')

# Enhanced text cleaning with lemmatization
logger.info("Initializing lemmatizer")
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text) if not isinstance(text, str) else text
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower().strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

logger.info("Cleaning text data")
tqdm.pandas()
df['combined_text'] = df['combined_text'].progress_apply(clean_text)
logger.info("Text cleaning completed")

# Step 3: Prepare features and target
logger.info("Preparing features and target for regression")
X = df[['combined_text', 'text_length', 'api_count', 'bug_count', 'feature_count', 'urgent_count']]
y = df['Estimated Hours']

# Scale target variable
logger.info("Scaling Estimated Hours")
scaler = StandardScaler()
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
joblib.dump(scaler, 'hours_scaler.pkl')
logger.info("Target scaling completed")

# Split data
logger.info("Splitting data into training and testing sets")
X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)
logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

# Step 4: Create and train the model pipeline
logger.info("Creating regression model pipeline with XGBoost")
pipeline = Pipeline([
    ('features', ColumnTransformer([
        ('text', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2)), 'combined_text'),
        ('numeric', StandardScaler(), ['text_length', 'api_count', 'bug_count', 'feature_count', 'urgent_count'])
    ])),
    ('regressor', xgb.XGBRegressor(random_state=42, verbosity=1, n_jobs=-1))
])

# Expanded hyperparameter tuning
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [3, 6, 9],
    'regressor__learning_rate': [0.01, 0.1],
    'regressor__subsample': [0.8, 1.0],
    'regressor__colsample_bytree': [0.8, 1.0]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
logger.info("Starting regression model training with grid search")
grid_search.fit(X_train, y_train)
logger.info("Regression model training completed")

# Best model
best_model = grid_search.best_estimator_
logger.info(f"Best regression parameters: {grid_search.best_params_}")

# Cross-validation scores
logger.info("Computing cross-validation scores")
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
logger.info(f"Cross-validation RMSE: Mean={cv_rmse.mean():.2f}, Std={cv_rmse.std():.2f}")

# Step 5: Evaluate the regression model
logger.info("Evaluating regression model")
y_pred_scaled = best_model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

# Inverse log transformation
y_pred = np.expm1(y_pred)
y_test_original = np.expm1(y_test_original)

mse = mean_squared_error(y_test_original, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_original, y_pred)
r2 = r2_score(y_test_original, y_pred)
logger.info(f"Regression Metrics:")
logger.info(f"Mean Squared Error (MSE): {mse:.2f}")
logger.info(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
logger.info(f"Mean Absolute Error (MAE): {mae:.2f}")
logger.info(f"R² Score: {r2:.2f}")

# Feature importance analysis
logger.info("Analyzing feature importance")
feature_names = best_model.named_steps['features'].get_feature_names_out()
importances = best_model.named_steps['regressor'].feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
logger.info(f"Top 5 features:\n{importance_df.head().to_dict()}")

# Step 6: Save the model and scaler
logger.info("Saving regression model to hours_prediction_model.pkl")
joblib.dump(best_model, 'hours_prediction_model.pkl')
logger.info("Regression model saved successfully")