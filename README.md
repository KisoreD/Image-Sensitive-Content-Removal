"# Image-Sensitive-Content-Removal" 
AI Image Classification – Unsafe Image Remover

📌 Overview
This project automatically classifies images as "Safe" or "Unsafe" using a pre-trained ResNet-50 model and removes images that are categorized as unsafe. It is useful for filtering out inappropriate or unwanted content from datasets.

🛠 Features
✅ Uses Microsoft’s ResNet-50 model for image classification
✅ Automatically detects and removes unsafe images
✅ Works with bulk image processing in a specified folder
✅ Built using PyTorch, Hugging Face Transformers, and TorchVision

📂 Dataset
The model is trained and tested using the UnsafeBench dataset, which contains labeled images categorized as "Safe" or "Unsafe."

🚀 Installation & Setup

1️⃣ Clone the Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2️⃣ Install Dependencies
Ensure you have Python 3.8+ installed. Then, install the required libraries:
pip install torch torchvision transformers datasets huggingface_hub pillow

3️⃣ Set Up Hugging Face Authentication
Since this project fetches a model from Hugging Face, authenticate using your 

Hugging Face token:
from huggingface_hub import login
login(token="your_huggingface_token")
🔹 Tip: Store the token securely instead of hardcoding it.

🖼 Usage

Prepare a folder containing the images to be processed.
Modify the folder_path in the script to point to your image directory.

Run the script:
python Aigetai_pro.py

The script will classify and remove unsafe images automatically.

📜 Code Explanation
🔹 Loads the ResNet-50 model for image classification.
🔹 Preprocesses images using TorchVision transformations.
🔹 Classifies each image as "Safe" or "Unsafe".
🔹 Deletes images classified as "Unsafe" and logs actions.

📌 Example Output

Kept: image1.jpg
Removed: image2.jpg
Kept: image3.jpg
Removed: image4.jpg

🛠 Future Improvements
✅ Improve classification accuracy using fine-tuning
✅ Add support for different pre-trained models
✅ Allow saving classified images into separate folders instead of deletion

📜 License
This project is open-source and available under the MIT License.

