import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from PIL import Image
import numpy as np
import joblib

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIM = 1280
# Placeholder for your trained classifier path
CLASSIFIER_PATH = "../results/models/image_only_classifier.pkl"

# Global model
IMAGE_MODEL = None
IMAGE_CLASSIFIER = None


def load_image_baseline_models():
    """Loads the EfficientNet-B0 model and the trained classifier."""
    global IMAGE_MODEL, IMAGE_CLASSIFIER

    if IMAGE_MODEL is None:
        IMAGE_MODEL = EfficientNet.from_pretrained('efficientnet-b0').to(DEVICE)
        IMAGE_MODEL._fc = nn.Identity()
        IMAGE_MODEL.eval()

    if IMAGE_CLASSIFIER is None:
        try:
            # You must train and save this classifier (e.g., Logistic Regression or MLP)
            IMAGE_CLASSIFIER = joblib.load(CLASSIFIER_PATH)
        except:
            print(f"Warning: Image classifier not found at {CLASSIFIER_PATH}. Using mock function.")

            # --- MOCK CLASSIFIER FOR DEMONSTRATION ---
            class MockImageClassifier:
                def predict_proba(self, X):
                    # Mock: slightly less confident prediction
                    real_prob = np.clip(0.5 + np.mean(X[0, :10]) * 0.001, 0.4, 0.7)
                    return np.array([[1 - real_prob, real_prob]])

            IMAGE_CLASSIFIER = MockImageClassifier()
            # ----------------------------------------

    return True


def predict_image_only(image: Image.Image):
    """
    Runs the Image-Only (EfficientNet) pipeline.

    Returns: prediction (str), confidence (float), real_prob (float)
    """
    if not load_image_baseline_models() or image is None:
        return "ERROR", 0.0, 0.0

    try:
        # 1. Feature Extraction (EfficientNet)
        image = image.convert("RGB").resize((224, 224))
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = (image_tensor - 0.5) / 0.5
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feature = IMAGE_MODEL(image_tensor).cpu().numpy()  # [1, 1280]

        # 2. Prediction
        proba = IMAGE_CLASSIFIER.predict_proba(feature)[0]
        real_prob = proba[1]

        prediction = "REAL" if real_prob > 0.5 else "FAKE"
        confidence = real_prob if prediction == "REAL" else proba[0]

        return prediction, confidence, real_prob

    except Exception as e:
        print(f"Image-Only Inference error: {e}")
        return "INFERENCE FAILED", 0.0, 0.0