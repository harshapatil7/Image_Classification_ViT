import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from evaluate import load
import numpy as np

# --- 1. Configuration and Setup ---
# Define your paths
TRAIN_LABELS_CSV = r"D:\image classification\Gemini\train_labels.csv"
NON_TEXTED_IMAGE_PATH = r"D:\image classification\Gemini\TrainData\Data\NoText"
TEXTED_IMAGE_PATH = r"D:\image classification\Gemini\TrainData\Data\Text"
TEST_IMAGES_PATH = r"D:\image classification\Gemini\TestData\TestData"

# Model configuration
MODEL_NAME = "google/vit-base-patch16-224-in21k" # A good general-purpose ViT model
NUM_LABELS = 2 # 0 for non-texted, 1 for texted
ID2LABEL = {0: "non-texted", 1: "texted"}
LABEL2ID = {"non-texted": 0, "texted": 1}

# Training arguments
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100
SAVE_STEPS = 500
EVAL_STEPS = 500
OUTPUT_DIR = "./vit_text_classification_model"

# Determine the device to use (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")


# --- 2. Load and Prepare Data ---

# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, df, img_dir_notext, img_dir_text, transform=None):
        self.df = df
        self.img_dir_notext = img_dir_notext
        self.img_dir_text = img_dir_text
        self.transform = transform

        self.image_paths = []
        self.labels = []
        for index, row in df.iterrows():
            image_name = row['id']
            label = row['label']
            
            # Determine the correct image path based on the label
            if label == 0: # Non-texted
                image_path = os.path.join(self.img_dir_notext, image_name)
            elif label == 1: # Texted
                image_path = os.path.join(self.img_dir_text, image_name)
            else:
                raise ValueError(f"Unknown label: {label} for image {image_name}")
            
            if os.path.exists(image_path):
                self.image_paths.append(image_path)
                self.labels.append(label)
            else:
                print(f"Warning: Image not found at {image_path}. Skipping.")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB") # Ensure 3 channels
        label = self.labels[idx]

        if self.transform:
            image = self.transform(images=image, return_tensors="pt")

        return {"pixel_values": image["pixel_values"].squeeze(), "labels": torch.tensor(label)}

# Load the labels CSV
try:
    labels_df = pd.read_csv(TRAIN_LABELS_CSV)
except FileNotFoundError:
    print(f"Error: train_labels.csv not found at {TRAIN_LABELS_CSV}")
    exit()

# Split data into training and validation sets
train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=42, stratify=labels_df['label'])

# --- 3. Define Image Preprocessing ---
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

# Create datasets
train_dataset = CustomImageDataset(train_df, NON_TEXTED_IMAGE_PATH, TEXTED_IMAGE_PATH, transform=processor)
val_dataset = CustomImageDataset(val_df, NON_TEXTED_IMAGE_PATH, TEXTED_IMAGE_PATH, transform=processor)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")

# --- 4. Load Pre-trained ViT Model ---
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)
model.to(device) # <--- Move the model to the selected device (GPU if available)

# --- 5. Train the Model ---

# Define collate function (needed for DataLoader)
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

# Define metrics
metric = load("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    logging_dir='./logs',
    logging_steps=LOGGING_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none", # You can set this to "tensorboard" or "wandb" for logging
    # The following line is crucial for enabling GPU with the Trainer,
    # though it's often inferred if `model.to(device)` is called and device is 'cuda'
    # It explicitly tells the Trainer to use CUDA if available.
    fp16=torch.cuda.is_available(), # Use mixed precision training if GPU is available (faster, less memory)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor, # Pass processor as tokenizer for ViT
)

print("Starting training...")
train_results = trainer.train()
print("Training complete.")

# Save the model
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR) # Save the processor too


# --- 6. Evaluate the Model (on validation set after training) ---
print("Evaluating the model on the validation set...")
eval_results = trainer.evaluate()
print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
print(eval_results)


# --- 7. Test with Custom Images ---
print("\n--- Testing with custom images ---")

# Load the trained model and processor
loaded_processor = ViTImageProcessor.from_pretrained(OUTPUT_DIR)
loaded_model = ViTForImageClassification.from_pretrained(OUTPUT_DIR)
loaded_model.eval() # Set to evaluation mode
loaded_model.to(device) # <--- Ensure the loaded model is also on the device

def predict_single_image(image_path, model, processor, device):
    if not os.path.exists(image_path):
        print(f"Error: Test image not found at {image_path}")
        return None

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()} # <--- Move input tensors to GPU

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]
    
    # Get probabilities (optional)
    probabilities = torch.softmax(logits, dim=1)[0].tolist()
    
    return predicted_label, probabilities

# Get all test image paths
test_image_files = [os.path.join(TEST_IMAGES_PATH, f) for f in os.listdir(TEST_IMAGES_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

if not test_image_files:
    print(f"No test images found in {TEST_IMAGES_PATH}. Please ensure images are in the specified folder.")
else:
    print(f"Found {len(test_image_files)} test images.")
    for i, img_path in enumerate(test_image_files[:5]): # Predict on first 5 for brevity
        print(f"\nPredicting for image: {os.path.basename(img_path)}")
        prediction, probs = predict_single_image(img_path, loaded_model, loaded_processor, device)
        if prediction is not None:
            print(f"Predicted Label: {prediction}")
            print(f"Probabilities (non-texted, texted): {probs}")
        if i >= 4: # Limit to 5 examples
            break
    print("\nPrediction on selected test images complete. You can modify the loop to predict on all test images.")