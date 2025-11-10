import torch
import joblib
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Placeholder for the trained LR model path
MODEL_PATH = "../results/models/clip_lr_model.pkl"

# Global variables to load models only once
CLIP_MODEL = None
CLIP_PROCESSOR = None
LR_MODEL = None


def load_clip_baseline_models():
    """Loads all necessary models (CLIP and the trained LR classifier) once."""
    global CLIP_MODEL, CLIP_PROCESSOR, LR_MODEL

    if CLIP_MODEL is None:
        CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if LR_MODEL is None:
        try:
            # You must train and save this classifier
            LR_MODEL = joblib.load(MODEL_PATH)
        except:
            print(f"Warning: CLIP LR model not found at {MODEL_PATH}. Using mock function.")

            # --- MOCK CLASSIFIER FOR DEMONSTRATION ---
            class MockLR:
                def predict_proba(self, X):
                    sim = X[0, 0]
                    # Mock: high sim favors REAL
                    real_prob = np.clip(0.5 + sim * 0.3, 0.1, 0.9)
                    return np.array([[1 - real_prob, real_prob]])

            LR_MODEL = MockLR()
            # ----------------------------------------

    return True


def predict_clip_similarity(text: str, image: Image.Image):
    """
    Performs the full CLIP Similarity inference pipeline.

    Returns: prediction (str), confidence (float), similarity (float)
    """
    if not load_clip_baseline_models() or not text or image is None:
        return "ERROR", 0.0, 0.0

    try:
        # 1. Calculate the single feature (Similarity)
        inputs = CLIP_PROCESSOR(text=[text], images=image, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            out = CLIP_MODEL(**inputs)
            img_emb = out.image_embeds.squeeze(0)
            txt_emb = out.text_embeds.squeeze(0)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            similarity = (img_emb * txt_emb).sum().cpu().item()

        # 2. Predict using LR classifier
        X = np.array([[similarity]])
        proba = LR_MODEL.predict_proba(X)[0]
        real_prob = proba[1]

        prediction = "REAL" if real_prob > 0.5 else "FAKE"
        confidence = real_prob if prediction == "REAL" else proba[0]

        return prediction, confidence, similarity

    except Exception as e:
        print(f"CLIP Inference error: {e}")
        return "INFERENCE FAILED", 0.0, 0.0