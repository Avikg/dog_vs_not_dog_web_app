import os
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for
from torchvision import transforms
from PIL import Image
from werkzeug.utils import secure_filename

# ================================
# 1. Flask App Configuration
# ================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Directory to store uploaded images

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ================================
# 2. Load Trained Model
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the BinaryViT model
model = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("binaryvit_dog_vs_not_dog_best.pth", map_location=device))
model.to(device)
model.eval()

print("✅ Model loaded successfully.")

# ================================
# 3. Define Preprocessing Function
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ================================
# 4. Flask Routes
# ================================

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            return "❌ No file uploaded", 400
        file = request.files['file']
        
        if file.filename == '':
            return "❌ No selected file", 400
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Run model inference
            prediction, attention_map_path = predict_image(file_path)

            return render_template('index.html', image_path=file_path, prediction=prediction, attention_map_path=attention_map_path)
    
    return render_template('index.html', image_path=None, prediction=None, attention_map_path=None)

# ================================
# 5. Prediction Function
# ================================

def predict_image(image_path):
    """Processes the uploaded image and runs model inference."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        label = "Dog" if predicted_class == 1 else "Not Dog"

    # Generate Attention Map
    attention_map_path = generate_attention_map(image_tensor, image_path)

    return label, attention_map_path

# ================================
# 6. Attention Map Function
# ================================

def generate_attention_map(image_tensor, image_path):
    """Extracts and visualizes the attention map from the model."""
    attention_maps = []

    def hook_fn(module, input, output):
        """Hook function to extract attention maps from self-attention layers."""
        attention_maps.append(output.detach().cpu())

    # Find the last self-attention layer and register a hook
    for name, module in model.named_modules():
        if "attn.proj" in name:
            module.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(image_tensor)

    if attention_maps:
        attn = attention_maps[-1]
        attn = attn.mean(dim=1).squeeze(0).numpy()

        # Reshape for visualization (DeiT uses 14x14 patches)
        attn = attn[:196].reshape(14, 14)

        # Save attention map
        plt.figure(figsize=(8, 8))
        sns.heatmap(attn, cmap="viridis", alpha=0.5, square=True)
        attention_map_path = os.path.join(app.config['UPLOAD_FOLDER'], "attention_map.png")
        plt.savefig(attention_map_path)
        plt.close()

        return attention_map_path

    return None

# ================================
# 7. Run Flask App
# ================================

if __name__ == '__main__':
    app.run(debug=True)
