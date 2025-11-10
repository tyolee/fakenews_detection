# importing helper function from src directory
import sys, os

sys.path.append(os.path.join(os.path.dirname(__name__), '..', 'src'))

import gradio as gr
from PIL import Image
from typing import Optional

# Import prediction functions from all baseline scripts
from baseline_text import predict_text_only
from baseline_image import predict_image_only
from baseline_clip import predict_clip_similarity


# --- Gradio Interface Setup ---

def run_prediction(model_choice: str, title: str, image: Optional[Image.Image]):
    """Interface wrapper function to route input to the selected model."""
    if not title or image is None:
        return "Please provide both a post title and an image.", "N/A", "N/A", "N/A", "N/A"

    # Default placeholder output
    pred_text, metric_score, interpretability = "N/A", "N/A", "N/A"

    if model_choice == "Baseline (Text-Only)":
        prediction, confidence, real_prob = predict_text_only(title)

        pred_text = f"TEXT-ONLY: {prediction} (Confidence: {confidence:.2%})"
        metric_score = f"{real_prob:.3f}"

        # Interpretability
        interpretability = (
            f"**Primary Metric:** Text-Only Probability of REAL ({real_prob:.3f})\n"
            f"**Analysis:** This prediction is based solely on the linguistic cues extracted by DistilBERT (e.g., tone, sensationalism)."
        )

    elif model_choice == "Baseline (Image-Only)":
        prediction, confidence, real_prob = predict_image_only(image)

        pred_text = f"IMAGE-ONLY: {prediction} (Confidence: {confidence:.2%})"
        metric_score = f"{real_prob:.3f}"

        # Interpretability
        interpretability = (
            f"**Primary Metric:** Image-Only Probability of REAL ({real_prob:.3f})\n"
            f"**Analysis:** This prediction is based solely on the visual features extracted by EfficientNet (e.g., image quality, content type, lighting)."
        )

    elif model_choice == "Baseline (CLIP Similarity)":
        prediction, confidence, similarity = predict_clip_similarity(title, image)

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
            f"**Primary Metric:** CLIP Cosine Similarity Score ({similarity:.3f})\n"
            f"**Analysis:** {sim_desc}. The model is checking if the image *looks like* what the text describes."
        )

    return pred_text, metric_score, interpretability


# Define the Gradio Interface components and layout
with gr.Blocks(title="Fakeddit Multimodal Detector Prototype") as demo:
    gr.Markdown("# 📰 Fakeddit Multimodal Detector Prototype")
    gr.Markdown("## Deliverable 2: Unimodal and Cross-Modal Baseline Implementation")

    model_selector = gr.Radio(
        ["Baseline (Text-Only)", "Baseline (Image-Only)", "Baseline (CLIP Similarity)"],
        label="Select Model Attempt",
        value="Baseline (CLIP Similarity)"
    )

    with gr.Row():
        # Input column
        with gr.Column(scale=1):
            title_input = gr.Textbox(lines=2, label="Post Title (Text Input)",
                                     placeholder="Enter the headline or post title here.")
            image_input = gr.Image(type="pil", label="Image Input", sources=['upload', 'clipboard'], height=250)
            predict_button = gr.Button("Analyze Post")

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

if __name__ == '__main__':
    # Attempt to load models for all pipelines

    # importing helper function from src directory
    from baseline_text import load_text_baseline_models
    from baseline_image import load_image_baseline_models
    from baseline_clip import load_clip_baseline_models

    load_text_baseline_models()
    load_image_baseline_models()
    load_clip_baseline_models()

    demo.launch()
