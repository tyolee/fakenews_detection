# Multimodal Fake News Detector on Fakeddit

## Project Title
Multimodal Fake News Detector

## Project Overview
This project develops a robust system for detecting fake news by analyzing both textual content and associated images from the Fakeddit dataset. We implement and benchmark several machine learning architectures, starting with unimodal baselines and culminating in a Deep Fusion model to capture complex cross-modal inconsistencies.

The system is built on the large-scale Fakeddit dataset.

## Installation and Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/tyolee/fakenews_detection
    ```

2.  **Setup Python Environment:**
    We recommend using a virtual environment (e.g., conda or venv).
    ```bash
    # Create and activate environment (example using conda)
    conda env create -f environment.yml
    conda activate AMLPRJ
    ```

3.  **Download Dataset:**
    * Place the three Fakeddit TSV files (`multimodal_train.tsv`, `multimodal_validate.tsv`, `multimodal_test_public.tsv`) inside the `data/` directory.
    * https://www.kaggle.com/code/vanshikavmittal/fakeddit-multimodal-fake-news-classification/input

## How to Run Your Notebook

1.  **Start JupyterLab/Notebook:**
    ```bash
    jupyter-lab
    ```
2.  **Execute `notebooks/setup.ipynb`:**
    * This notebook performs crucial steps: dataset loading, stratified sampling, and the image download/filtering process.
    * **CRITICAL:** Run this notebook first to create the clean, local image files necessary for all subsequent image/multimodal experiments.

3. **Model Training**
    **Execute `notebooks/baseline_training.ipynb`:**
    

5. **Gradio Interface**
    ```bash
    python fakenews_gradio.py
    ```

## System Architecture 
The pipeline is modular, separating data handling, feature extraction, and classification:

1. Preprocessing: Tokenization for text, resizing/normalization for images, and robust filtering for corrupt image links.

2. Feature Extraction: Separate deep learning models (DistilBERT, EfficientNet, CLIP) are used to generate high-dimensional feature vectors.

3. Classification: Using simple classifiers for baselines and a sophisticated Multi-Layer Perceptron (MLP) for the Deep Fusion approach.


### Feature Extrators:
* Text: DistilBERT (768 dimensions)
* Image: EfficientNet-B0 (1280 dimensions)
* Cross-Modal Alignment: CLIP (Similarity Score, 1 dimension)
* The Deep Fusion architecture concatenates all three feature vectors (Total dimension: 2049) before passing them through a series of dense layers for final classification.

## Dataset Information
* **Name:** Fakeddit
* **Modality:** Multimodal (Text, Image)
* **Label:** 2-way classification (0: Fake, 1: Real)
* **Size:** $\sim 680,000$ total samples (sampled subset is used for training/evaluation).

## Project Notebooks
* setup.ipynb: Handles dataset loading, filtering, stratified sampling, and image downloading.
* baseline_training.ipynb: Extracts features and trains the three unimodal/cross-modal baseline classifiers (Text-Only, Image-Only, CLIP Similarity).
* fusion_training.ipynb: (Current Focus) Extracts all features and trains the Deep Fusion MLP model using the combined 2049-dimension feature vector.

# User Interface Concept
The final application will present an interactive interface for real-time fake news analysis, showing the confidence scores from the individual baselines alongside the final prediction from the Deep Fusion model.
<img width="1676" height="574" alt="image" src="https://github.com/user-attachments/assets/80e2c9d7-0f0f-42bf-bfcb-3c22e26107d3" />


## Author Name and Contact
Yoonki Lee(yoonki.lee@ufl.edu)
