import torch
import os
import shutil
from datasets import load_dataset
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from torchvision import transforms

from huggingface_hub import login
login(token="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")

from PIL import Image
import requests
ds = load_dataset("yiting/UnsafeBench")
label_mapping = {"Safe": 0, "Unsafe": 1}
model_name = "microsoft/resnet-50"
model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    return "Safe" if predicted_class == 0 else "Unsafe"

def process_images(folder_path):
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        label = classify_image(image_path)
        
        if label == "Unsafe":
            os.remove(image_path) 
            print(f"Removed: {image_file}")
        else:
            print(f"Kept: {image_file}")
folder_path = "D:\output_image"  
process_images(folder_path)
