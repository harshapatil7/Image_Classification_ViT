# Image_Classification_ViT

An advanced binary image classification pipeline leveraging Vision Transformers (ViT) to distinguish between images containing text ("texted") and those without ("non-texted").

This project moves beyond traditional Convolutional Neural Networks (CNNs) by utilizing global self-attention mechanisms to capture long-range spatial dependencies in media assets.

## 🚀 Key Features

   * `Transformer-Based Architecture`: Utilizes the google/vit-base-patch16-224-in21k model, which treats image patches as tokens, allowing for a global receptive field from the first layer.

   * `Custom Data Engineering`: Implements a robust PyTorch Dataset class with stratified sampling to ensure balanced class representation during training.

   * `Optimized Training`: Employs Mixed-Precision (FP16) and AdamW weight decay to maximize GPU throughput and ensure training stability.

   * `Interactive Inference`: Includes a dedicated CLI tool for real-time classification of custom image files.

## 📁 Project Structure

```
.
├── TrainData/                       # Training images (NoText/ and Text/ subfolders)
├── TestData/                        # Unlabeled images for final testing
├── vit_text_classification_model/   # Saved model weights and processor configuration
├── Image_Classifier.py              # Main training and evaluation script
├── predict_custom_image.py          # CLI script for single-image predictions
├── train_labels.csv                 # Mapping of image IDs to labels (0 or 1)
├── sample_submission.csv            # Format for Kaggle competition submission
└── photo.jpg / photo2.jpg           # Sample images for testing
```

## 🛠️ Installation & Setup

   `Environment`: Ensure you have Python 3.8+ and a CUDA-enabled GPU for optimal performance.

   `Dependencies`:

```
pip install torch torchvision pandas scikit-learn transformers evaluate pillow numpy 
```

   `Data Source`: Download the dataset from Kaggle: Text or No Text.

## 🏋️ Training the Model

The Main.py script handles the end-to-end pipeline: data loading, preprocessing via ViTImageProcessor, training using the Hugging Face Trainer API, and validation.

### Technical Specifications:

   `Input Resolution`: 224×224 pixels.

   `Patch Size`: 16×16.

   `Batch Size`: 16.

   `Optimizer`: AdamW with a Learning Rate of 2×10−5.

## To start training:
```
python Main.py
```
## 🔍 Inference & Prediction

Once the model is trained and saved in the ./vit_text_classification_model directory, you can classify any image using the interactive script.
Bash
```
python predict_custom_image.py
```
## How it works:

    The script loads the trained ViT model and processor.

    Provide the full path to an image (e.g., C:/Downloads/photo.jpg).

    The model outputs the predicted label along with the raw logits and class probabilities.

    Note: The model uses id2label mapping to return human-readable results: texted or non-texted.

## 📊 Evaluation Results

The model is evaluated on a 20% validation split. The training pipeline uses Accuracy as the primary metric to determine the "Best Model" for saving.
```
Metric	Value
Model	ViT-base-patch16-224
Epochs	3
Precision	Mixed (FP16)
Validation Accuracy	(Check your terminal output for final %)
```

## 📜 Credits

    Dataset: Kaggle [Text-or-No-Text](https://www.kaggle.com/competitions/text-or-no-text/data).

    Architecture: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)