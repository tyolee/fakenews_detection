# Multimodal Fake News Detector on Fakeddit

## Project Title
Multimodal Fake News Detector

## Project Overview
This project aims to develop a robust system for detecting misinformation (fake news) in social media posts that combine both textual headlines and associated imagery. The goal is to move beyond traditional unimodal detection methods by implementing an advanced deep learning Deep Fusion Architecture that assesses both the veracity of each modality and the semantic consistency between them.

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

3. Classification: Currently handled by a lightweight Logistic Regression model.

4. Deep Fusion (Next Step): Features will be concatenated and passed through a complex fusion network with an attention mechanism.


## Dataset Information
* **Name:** Fakeddit
* **Modality:** Multimodal (Text, Image)
* **Label:** 2-way classification (0: Fake, 1: Real)
* **Size:** $\sim 680,000$ total samples (sampled subset is used for training/evaluation).



## Author Name and Contact
Yoonki Lee(yoonki.lee@ufl.edu)
