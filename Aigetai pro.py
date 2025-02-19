import torch
import os
import shutil
from datasets import load_dataset
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from torchvision import transforms
from huggingface_hub import login
from PIL import Image
import requests

# ------------------------------
# Login to Hugging Face Hub
# ------------------------------
# Note: Store your token securely instead of hardcoding it here
login(token="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")

# ------------------------------
# Load Dataset
# ------------------------------
# Load the UnsafeBench dataset containing labeled 'Safe' and 'Unsafe' images
# This dataset is used for benchmarking unsafe image classification

ds = load_dataset("yiting/UnsafeBench")

# ------------------------------
# Define Label Mapping
# ------------------------------
# Mapping categories to numerical labels for classification
label_mapping = {"Safe": 0, "Unsafe": 1}

# ------------------------------
# Load Pre-trained Model
# ------------------------------
# Using Microsoft ResNet-50 model, pre-trained on image classification tasks
model_name = "microsoft/resnet-50"
model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)

# ------------------------------
# Load Feature Extractor
# ------------------------------
# The feature extractor preprocesses images before passing them into the model
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# ------------------------------
# Define Image Preprocessing Pipeline
# ------------------------------
# Resizing images to 224x224 pixels and converting them to tensor format
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Rescale image size
    transforms.ToTensor(),          # Convert image to tensor format
])

def classify_image(image_path):
    """
    Classify an image as 'Safe' or 'Unsafe'.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        str: Classification result ('Safe' or 'Unsafe').
    """
    # Load image and convert it to RGB format
    image = Image.open(image_path).convert("RGB")  
    
    # Apply necessary transformations and prepare for model input
    image = transform(image).unsqueeze(0)  
    
    # Perform inference without gradient calculation for efficiency
    with torch.no_grad():  
        outputs = model(image)  # Get model predictions
        logits = outputs.logits  # Extract classification logits
        predicted_class = torch.argmax(logits, dim=1).item()  # Get predicted label

    return "Safe" if predicted_class == 0 else "Unsafe"  # Convert prediction to label

def process_images(folder_path):
    """
    Process all images in the specified folder, classifying them and removing 'Unsafe' images.
    
    Args:
        folder_path (str): Path to the directory containing images.
    """
    # Iterate through each image file in the specified folder
    for image_file in os.listdir(folder_path):  
        image_path = os.path.join(folder_path, image_file)  # Construct full file path
        label = classify_image(image_path)  # Perform classification
        
        if label == "Unsafe":
            os.remove(image_path)  # Delete unsafe images
            print(f"Removed: {image_file}")  # Log removal action
        else:
            print(f"Kept: {image_file}")  # Log kept images

# ------------------------------
# Run Image Processing
# ------------------------------
# Define folder path containing images (Ensure to modify this path as needed)
folder_path = "D:\\output_image"  
process_images(folder_path)  # Execute image classification and filtering
