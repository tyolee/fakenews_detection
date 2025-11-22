import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from efficientnet_pytorch import EfficientNet
from PIL import Image
import numpy as np
from typing import Optional
import time

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_DIM = 768  # DistilBERT output dim
IMAGE_DIM = 1280  # EfficientNet-B0 output dim
FUSION_DIM = TEXT_DIM + IMAGE_DIM  # 2048

# CRITICAL FIX: The saved model has a single output (1 logit), not 2 classes.
# We must adjust the architecture to match the saved weights.
OUTPUT_LOGITS = 1

# Path to the PyTorch model weights from fusion_training.ipynb
MODEL_PATH = "../results/models/deep_fusion_model.pth"

# Global models (loaded once)
TEXT_MODEL = None
TEXT_TOKENIZER = None
IMAGE_MODEL = None
DFN_MODEL = None


# --- Deep Fusion Network Architecture (Adjusted) ---

class DeepFusionNet(nn.Module):
    """
    Multilayer Perceptron (MLP) for binary classification,
    outputting a single logit that represents the score for the positive class (REAL).
    """

    def __init__(self, input_dim):
        super(DeepFusionNet, self).__init__()
        # Fusion Layer: Input 2048 features, reduce to 512
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout = nn.Dropout(0.3)
        # Output Layer: 512 features to 1 logit (score for the positive class)
        self.fc2 = nn.Linear(512, OUTPUT_LOGITS)

    def forward(self, x):
        #
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# --- Model Loading ---

def load_fusion_models() -> bool:
    """
    Loads all necessary models for the Deep Fusion Network (DFN):
    1. DistilBERT for Text Features.
    2. EfficientNet-B0 for Image Features.
    3. The trained DeepFusionNet (PyTorch weights).
    """
    global TEXT_MODEL, TEXT_TOKENIZER, IMAGE_MODEL, DFN_MODEL

    # 1. Load Text Feature Extractor (DistilBERT)
    if TEXT_MODEL is None:
        try:
            print("Loading DFN Text Feature Extractor (DistilBERT)...")
            TEXT_MODEL = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
            TEXT_TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            TEXT_MODEL.eval()
        except Exception as e:
            print(f"Error loading DistilBERT: {e}")
            return False

    # 2. Load Image Feature Extractor (EfficientNet-B0)
    if IMAGE_MODEL is None:
        try:
            print("Loading DFN Image Feature Extractor (EfficientNet-B0)...")
            IMAGE_MODEL = EfficientNet.from_pretrained('efficientnet-b0').to(DEVICE)
            IMAGE_MODEL._fc = nn.Identity()  # Remove the final classification layer
            IMAGE_MODEL.eval()
        except Exception as e:
            print(f"Error loading EfficientNet: {e}")
            return False

    # 3. Load Fusion Classifier (DeepFusionNet)
    if DFN_MODEL is None:
        DFN_MODEL = DeepFusionNet(FUSION_DIM).to(DEVICE)
        try:
            print(f"Loading DFN PyTorch model weights from {MODEL_PATH}...")
            # Load weights from the .pth file
            DFN_MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            DFN_MODEL.eval()
            print("DeepFusionNet loaded successfully.")
        except FileNotFoundError:
            print(
                f"Warning: DFN PyTorch model not found at {MODEL_PATH}. Using untrained model (random weights) for mock prediction.")
            # Keep DFN_MODEL with random weights for mock inference
        except Exception as e:
            print(f"Error loading DFN PyTorch model: {e}")
            # If size mismatch error occurs again, it might indicate a different issue,
            # but we allow it to proceed with the mock model for demonstration.
            return False

    return True


# --- Prediction Function ---

def predict_fusion(text: str, image: Image.Image) -> tuple[str, float, float]:
    """
    Runs the Deep Fusion Network (DFN) pipeline.

    Returns: prediction (str), confidence (float), real_prob (float)
    """
    if not load_fusion_models() or not text or image is None:
        return "ERROR", 0.0, 0.0

    try:
        # --- 1. Feature Extraction (Text) ---
        inputs = TEXT_TOKENIZER(text, return_tensors="pt", truncation=True, padding='max_length', max_length=50).to(
            DEVICE)
        with torch.no_grad():
            outputs = TEXT_MODEL(**inputs)
            # Use the [CLS] token output as the text feature
            text_feature = outputs.last_hidden_state[:, 0, :]  # [1, 768]

        # --- 2. Feature Extraction (Image) ---
        image_resized = image.convert("RGB").resize((224, 224))
        image_tensor = torch.tensor(np.array(image_resized)).permute(2, 0, 1).float() / 255.0
        image_tensor = (image_tensor - 0.5) / 0.5
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_feature = IMAGE_MODEL(image_tensor)  # [1, 1280]

        # --- 3. Feature Concatenation (Deep Fusion) ---
        fused_feature = torch.cat([text_feature, image_feature], dim=1)  # [1, 2048]

        # --- 4. Final Prediction (DeepFusionNet) ---
        with torch.no_grad():
            logit = DFN_MODEL(fused_feature)  # Output is a single logit
            # Apply Sigmoid to the logit to get the probability of the positive class (REAL)
            real_prob = torch.sigmoid(logit).cpu().item()

            # Determine prediction and confidence
        if real_prob >= 0.5:
            prediction_label = "REAL"
            confidence = real_prob
        else:
            prediction_label = "FAKE"
            confidence = 1.0 - real_prob  # Confidence is always max(P(FAKE), P(REAL))

        return prediction_label, confidence, real_prob

    except Exception as e:
        print(f"DFN Prediction Error: {e}")
        # Fallback in case of runtime error
        return "PREDICTION_FAILED", 0.5, 0.5