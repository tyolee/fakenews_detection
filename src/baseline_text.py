import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import joblib
import numpy as np

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_DIM = 768  # DistilBERT output dim
# Placeholder for your trained classifier path
CLASSIFIER_PATH = "../results/models/text_only_classifier.pkl"

# Global models
TEXT_MODEL = None
TEXT_TOKENIZER = None
TEXT_CLASSIFIER = None


def load_text_baseline_models():
    """Loads the DistilBERT model, tokenizer, and the trained classifier."""
    global TEXT_MODEL, TEXT_TOKENIZER, TEXT_CLASSIFIER

    if TEXT_MODEL is None:
        TEXT_MODEL = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
        TEXT_TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        TEXT_MODEL.eval()

    if TEXT_CLASSIFIER is None:
        try:
            # You must train and save this classifier (e.g., Logistic Regression or MLP)
            TEXT_CLASSIFIER = joblib.load(CLASSIFIER_PATH)
        except:
            print(f"Warning: Text classifier not found at {CLASSIFIER_PATH}. Using mock function.")

            # --- MOCK CLASSIFIER FOR DEMONSTRATION ---
            class MockTextClassifier:
                def predict_proba(self, X):
                    # Mock: favors REAL if feature mean is positive
                    real_prob = np.clip(0.6 + np.mean(X) * 0.05, 0.45, 0.85)
                    return np.array([[1 - real_prob, real_prob]])

            TEXT_CLASSIFIER = MockTextClassifier()
            # ----------------------------------------

    return True


def predict_text_only(text: str):
    """
    Runs the Text-Only (DistilBERT) pipeline.

    Returns: prediction (str), confidence (float), real_prob (float)
    """
    if not load_text_baseline_models() or not text:
        return "ERROR", 0.0, 0.0

    try:
        # 1. Feature Extraction (DistilBERT)
        inputs = TEXT_TOKENIZER(text, return_tensors="pt", truncation=True, padding='max_length', max_length=50).to(
            DEVICE)
        with torch.no_grad():
            outputs = TEXT_MODEL(**inputs)
            feature = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [1, 768]

        # 2. Prediction
        proba = TEXT_CLASSIFIER.predict_proba(feature)[0]
        real_prob = proba[1]

        prediction = "REAL" if real_prob > 0.5 else "FAKE"
        confidence = real_prob if prediction == "REAL" else proba[0]

        return prediction, confidence, real_prob

    except Exception as e:
        print(f"Text-Only Inference error: {e}")
        return "INFERENCE FAILED", 0.0, 0.0