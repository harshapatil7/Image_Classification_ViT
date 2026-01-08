import torch
from PIL import Image
import os
from transformers import ViTForImageClassification, ViTImageProcessor

# --- Configuration ---
# This must match the OUTPUT_DIR where your model was saved after training
MODEL_SAVE_PATH = "./vit_text_classification_model"

# Determine the device to use (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for inference: {device}")
if device.type == 'cuda':
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Load the trained model and processor once 
try:
    processor = ViTImageProcessor.from_pretrained(MODEL_SAVE_PATH)
    model = ViTForImageClassification.from_pretrained(MODEL_SAVE_PATH)
    model.eval() # Set the model to evaluation mode
    model.to(device) # Move the model to the GPU
    print("\nModel and processor loaded successfully.")
except Exception as e:
    print(f"Error loading model or processor from '{MODEL_SAVE_PATH}': {e}")
    print("Please ensure the model directory exists and contains the saved model files.")
    exit() # Exit if model cannot be loaded

def predict_single_image(image_path, model, processor, device):
    """
    Predicts if a given image is 'texted' or 'non-texted' using the loaded model.

    Args:
        image_path (str): The full path to the custom image you want to classify.
        model (ViTForImageClassification): The loaded ViT model.
        processor (ViTImageProcessor): The loaded ViT image processor.
        device (torch.device): The device (CPU or GPU) to run inference on.

    Returns:
        tuple: A tuple containing:
            - str: The predicted label ('texted' or 'non-texted').
            - list: A list of probabilities for each class (e.g., [prob_non_texted, prob_texted]).
            - dict: The raw logits from the model.
        None: If the image path is invalid or prediction fails.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'")
        return None

    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB") # Ensure 3 channels
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()} # Move input tensors to GPU

        # Perform inference
        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = model(**inputs)

        # Get logits and probabilities
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0].tolist()

        # Get the predicted class label
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]

        return predicted_label, probabilities, logits.tolist()[0] # Return raw logits as a list for detail

    except Exception as e:
        print(f"An error occurred during prediction for '{image_path}': {e}")
        return None

if __name__ == "__main__":
    print("\n--- Custom Image Classification Loop ---")
    print("Enter the full path to an image, or type 'exit' to quit.")

    while True:
        custom_image_path = input("\nEnter image path (or 'exit'): ").strip()

        if custom_image_path.lower() == 'exit':
            print("Exiting image classification loop. Goodbye!")
            break
        
        if not custom_image_path: # Handle empty input
            print("No path entered. Please provide a path or type 'exit'.")
            continue

        prediction_result = predict_single_image(custom_image_path, model, processor, device)

        if prediction_result:
            predicted_label, probabilities, raw_logits = prediction_result
            print(f"  Image Path: {custom_image_path}")
            print(f"  Predicted Label: {predicted_label}")
            print(f"  Probabilities (non-texted, texted): {probabilities}")
            print(f"  Raw Logits: {raw_logits}")
        else:
            print("  Prediction failed for this image.")