# Multimodal Fake News Detector on Fakeddit

## Project Title
Multimodal Fake News Detector

## Brief Description
This project develops deep learning models for binary fake news detection (Fake/Real) using the Fakeddit dataset. We implement competitive unimodal baselines and a CLIP Similarity model deployed via an interactive Gradio interface that incorporates **CLIP similarity for prediction interpretability**.

## Installation and Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/tyolee/fakenews_detection
    ```

2.  **Setup Python Environment:**
    We recommend using a virtual environment (e.g., conda or venv).
    ```bash
    # Create and activate environment (example using conda)
    conda create -n fakeddit_env python=3.12
    conda activate fakeddit_env
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


## Dataset Information
* **Name:** Fakeddit
* **Modality:** Multimodal (Text, Image)
* **Label:** 2-way classification (0: Fake, 1: Real)
* **Size:** $\sim 680,000$ total samples (sampled subset is used for training/evaluation).

## Author Name and Contact
Yoonki Lee(yoonki.lee@ufl.edu)
