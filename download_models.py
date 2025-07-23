import os
import sys
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

# --- Configuration ---
# Corrected model names from the Hugging Face Hub
MODELS_TO_DOWNLOAD = {
    'bi_encoder': 'all-MiniLM-L6-v2',
    'cross_encoder': 'cross-encoder/ms-marco-MiniLM-L-6-v2', # <--- COMMA ADDED HERE
    'summarizer': 'sshleifer/distilbart-cnn-6-6'
}

# Local paths where the models will be saved
BASE_MODELS_PATH = './models'


def download_all_models():
    """
    Downloads and saves all required models.
    Checks if a model already exists before downloading.
    Handles errors for individual downloads without crashing.
    """
    print("--- Starting Model Download & Verification Process ---")
    
    os.makedirs(BASE_MODELS_PATH, exist_ok=True)
    
    all_successful = True

    # --- 1. Download and save the Bi-Encoder ---
    model_key = 'bi_encoder'
    model_name = MODELS_TO_DOWNLOAD[model_key]
    model_path = os.path.join(BASE_MODELS_PATH, model_key)
    print(f"\n[1/3] Checking for Bi-Encoder model ('{model_name}')...")
    
    if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, 'config.json')):
        print(f"      > Found existing model at '{model_path}'. Skipping download.")
    else:
        print(f"      > Model not found. Attempting to download '{model_name}'...")
        try:
            bi_encoder = SentenceTransformer(model_name)
            bi_encoder.save(model_path)
            print(f"      > SUCCESS: Bi-Encoder saved to '{model_path}'")
        except Exception as e:
            print(f"      > FAILED to download Bi-Encoder. Error: {e}")
            all_successful = False

    # --- 2. Download and save the Cross-Encoder ---
    model_key = 'cross_encoder'
    model_name = MODELS_TO_DOWNLOAD[model_key]
    model_path = os.path.join(BASE_MODELS_PATH, model_key)
    print(f"\n[2/3] Checking for Cross-Encoder model ('{model_name}')...")
    
    if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, 'config.json')):
        print(f"      > Found existing model at '{model_path}'. Skipping download.")
    else:
        print(f"      > Model not found. Attempting to download '{model_name}'...")
        try:
            cross_encoder = CrossEncoder(model_name)
            cross_encoder.save(model_path)
            print(f"      > SUCCESS: Cross-Encoder saved to '{model_path}'")
        except Exception as e:
            print(f"      > FAILED to download Cross-Encoder. Error: {e}")
            all_successful = False
            
    # --- 3. Download and save the Summarizer ---
    model_key = 'summarizer'
    model_name = MODELS_TO_DOWNLOAD[model_key]
    model_path = os.path.join(BASE_MODELS_PATH, model_key)
    print(f"\n[3/3] Checking for Summarizer model ('{model_name}')...")

    if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, 'config.json')):
        print(f"      > Found existing model at '{model_path}'. Skipping download.")
    else:
        print(f"      > Model not found. Attempting to download '{model_name}'...")
        try:
            summarizer = pipeline("summarization", model=model_name, tokenizer=model_name)
            summarizer.save_pretrained(model_path)
            print(f"      > SUCCESS: Summarizer saved to '{model_path}'")
        except Exception as e:
            print(f"      > FAILED to download Summarizer. Error: {e}")
            all_successful = False

    print("\n--- Process Finished ---")
    if all_successful:
        print("All models are downloaded and ready.")
    else:
        print("One or more models failed to download. Please check the error messages above and your internet connection.")
        sys.exit(1)


if __name__ == '__main__':
    download_all_models()