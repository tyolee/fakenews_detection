import sys, os

# importing helper function from src directory
sys.path.append(os.path.join(os.path.dirname(__name__), '..', 'src'))

import gradio as gr
from PIL import Image
from typing import Optional
import time

# Import prediction functions from all baseline scripts AND the new DFN model
from baseline_text import predict_text_only, load_text_baseline_models
from baseline_image import predict_image_only, load_image_baseline_models
from baseline_clip import predict_clip_similarity, load_clip_baseline_models
from fusion_model import predict_fusion, load_fusion_models  # Assumes load_fusion_models exists


# --- Gradio Interface Setup ---

def run_prediction(model_choice: str, title: str, image: Optional[Image.Image]):
    """Interface wrapper function to route input to the selected model."""
    if not title or image is None:
        return "Please provide both a post title and an image.", "N/A", "N/A", "N/A", "N/A"

    # Default placeholder output
    pred_text, metric_score, interpretability = "N/A", "N/A", "N/A"

    # Start timer for latency measurement
    start_time = time.time()

    try:
        if model_choice == "Baseline (Text-Only)":
            prediction, confidence, real_prob = predict_text_only(title)
            metric_label = "Probability of REAL"

            pred_text = f"TEXT-ONLY: {prediction} (Confidence: {confidence:.2%})"
            metric_score = f"{real_prob:.3f}"

            # Interpretability
            interpretability = (
                f"**Primary Metric:** {metric_label} ({real_prob:.3f})\n"
                f"**Analysis:** This prediction is based solely on the linguistic cues extracted by DistilBERT (e.g., tone, sensationalism)."
            )

        elif model_choice == "Baseline (Image-Only)":
            prediction, confidence, real_prob = predict_image_only(image)
            metric_label = "Probability of REAL"

            pred_text = f"IMAGE-ONLY: {prediction} (Confidence: {confidence:.2%})"
            metric_score = f"{real_prob:.3f}"

            # Interpretability
            interpretability = (
                f"**Primary Metric:** {metric_label} ({real_prob:.3f})\n"
                f"**Analysis:** This prediction is based solely on the visual features extracted by EfficientNet (e.g., image quality, content type, lighting)."
            )

        elif model_choice == "Baseline (CLIP Similarity)":
            prediction, confidence, similarity = predict_clip_similarity(title, image)
            metric_label = "CLIP Cosine Similarity Score"

            pred_text = f"CLIP SIMILARITY: {prediction} (Confidence: {confidence:.2%})"
            metric_score = f"{similarity:.3f}"

            # Interpretability
            if similarity > 0.6:
                sim_desc = "High Match: Text and image are semantically aligned."
            elif similarity < 0.2:
                sim_desc = "Low Match: Text and image are semantically disparate (Potential Miscontext)."
            else:
                sim_desc = "Neutral Match: Weak correlation between modalities."

            interpretability = (
                f"**Primary Metric:** {metric_label} ({similarity:.3f})\n"
                f"**Analysis:** {sim_desc}. The model checks the semantic coherence between the two modalities."
            )

        elif model_choice == "Deep Fusion Network (DFN)":
            prediction, confidence, real_prob = predict_fusion(title, image)
            metric_label = "Probability of REAL (DFN Output)"

            pred_text = f"DFN FUSION: {prediction} (Confidence: {confidence:.2%})"
            metric_score = f"{real_prob:.3f}"

            # Interpretability
            interpretability = (
                f"**Primary Metric:** {metric_label} ({real_prob:.3f})\n"
                f"**Analysis:** This is the most advanced model, combining features from both text (DistilBERT) and image (EfficientNet/ViT) using a non-linear fusion layer to capture complex cross-modal interactions. [Image of Deep Fusion Network Architecture] The result is a holistic assessment of the post's veracity."
            )

        else:
            pred_text = "ERROR: Invalid model selection."
            interpretability = "Please select a valid model from the radio buttons."

        end_time = time.time()
        latency = end_time - start_time

        # Append latency to interpretability
        interpretability += f"\n\n**Inference Time:** {latency:.2f} seconds."

    except Exception as e:
        pred_text = "PREDICTION FAILED"
        metric_score = "N/A"
        interpretability = f"An unexpected error occurred during model inference: {str(e)}"

    return pred_text, metric_score, interpretability


# Define the Gradio Interface components and layout
with gr.Blocks(title="Fakeddit Multimodal Detector Prototype") as demo:
    gr.Markdown("# Fakeddit Multimodal Detector Prototype")
    gr.Markdown("## Deliverable 3: Deep Fusion Network (DFN) Integration")

    model_selector = gr.Radio(
        [
            "Baseline (Text-Only)",
            "Baseline (Image-Only)",
            "Baseline (CLIP Similarity)",
            "Deep Fusion Network (DFN)"  # Added DFN
        ],
        label="Select Model Attempt",
        value="Deep Fusion Network (DFN)"  # Set DFN as default
    )

    with gr.Row():
        # Input column
        with gr.Column(scale=1):
            title_input = gr.Textbox(lines=2, label="Post Title (Text Input)",
                                     placeholder="Enter the headline or post title here.")
            image_input = gr.Image(type="pil", label="Image Input", sources=['upload', 'clipboard'], height=250)
            predict_button = gr.Button("Analyze Post", variant="primary")

        # Output column
        with gr.Column(scale=2):
            prediction_output = gr.Textbox(label="Final Prediction", lines=1, interactive=False, max_lines=1)
            metric_score_output = gr.Textbox(label="Model-Specific Metric Score", lines=1, interactive=False,
                                             max_lines=1)
            interpretability_output = gr.Markdown(label="Interpretability (Model Reasoning)")

    # Link the function to the button click event
    predict_button.click(
        fn=run_prediction,
        inputs=[model_selector, title_input, image_input],
        outputs=[prediction_output, metric_score_output, interpretability_output]
    )

    gr.Markdown("""
    ---
    **Note on Model Loading:** The first time a prediction is run for any model, it will take several seconds 
    as the underlying pre-trained models (DistilBERT, EfficientNet, CLIP) are loaded into memory.
    """)

if __name__ == '__main__':
    # Load all models once on startup
    print("Loading all models for all pipelines...")
    load_text_baseline_models()
    load_image_baseline_models()
    load_clip_baseline_models()
    # Assuming this function is implemented in fusion_model.py to load DFN components
    try:
        load_fusion_models()
        print("DFN model components loaded successfully.")
    except Exception as e:
        print(
            f"Warning: Could not load DFN models via load_fusion_models(). The DFN model might use mock prediction: {e}")

    print("All models initialized. Launching Gradio demo.")
    demo.launch()