# 🎮 Flask Web Application - Dog vs Not Dog Classification

This project implements a **Flask-based web application** for classifying images as **Dog or Not Dog** using a **Vision Transformer (ViT) - BinaryViT model**. The application allows users to upload an image, runs inference on the trained ViT model, and displays the classification along with an attention map highlighting important image regions.

---

## **📂 Project Structure**
```
/webapp/
│── static/                     # Contains uploaded images and generated attention maps
│   ├── uploads/               # Stores user-uploaded images
│── templates/                  # HTML files for the web UI
│   └── index.html            # Main page for image upload & result display
│── app.py                      # Flask backend application
│── binaryvit_dog_vs_not_dog_best.pth  # Trained BinaryViT model
│── requirements.txt            # Required Python libraries
│── README.md                    # This documentation
```

---

## **📅 Requirements**
Ensure you have the following dependencies installed before running the application.

### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

**Or install manually:**
```bash
pip install flask torch torchvision timm numpy pillow matplotlib seaborn werkzeug
```

---

## **🚀 Running the Flask Web App**
### **2️⃣ Start the Web Server**
Run the following command:
```bash
python app.py
```
By default, the app will be accessible at **`http://127.0.0.1:5000/`**.

---

## **👾 Features**
✅ **Upload an image** (PNG, JPG, JPEG)  
✅ **Binary classification:** "Dog" or "Not Dog"  
✅ **Attention map visualization** to highlight important image regions  
✅ **Runs on CPU or GPU** (uses CUDA if available)  
✅ **User-friendly Flask interface**  

---

## **🌍 Usage**
### **3️⃣ How to Use the Web App**
1. **Open the application** in your browser at `http://127.0.0.1:5000/`.
2. **Upload an image** using the form.
3. **Click "Upload & Predict"** to process the image.
4. **View the prediction** and **generated attention map**.

---

## **📷 Example Output**
- **User uploads an image** (e.g., `dog.jpg`).
- **Model classifies it as "Dog" or "Not Dog"**.
- **Displays the uploaded image**.
- **Shows an attention map overlay**.

### **Example UI:**
```
+--------------------------------+
| 🐶 Dog or Not Dog - Upload    |
| [ Choose File ]  (dog.jpg)     |
| [ Upload & Predict ]           |
+--------------------------------+

Prediction: ✅ "Dog"

🔥 Model Attention Map:
[Image with Attention Overlay]
```

---

## **🔧 Deployment Options**
You can deploy this Flask app on:
- **Local machine** (Run `python app.py`)
- **Docker** (Create a Dockerfile and containerize the app)
- **Cloud platforms** (AWS, Google Cloud, or Heroku)

---

## **📘 Research Papers Used**
### **1️⃣ Vision Transformer (ViT) - Foundational Paper**
**Title:** An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale  
**Authors:** Alexey Dosovitskiy et al.  
**Link:** [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)  

### **2️⃣ Data-Efficient Image Transformer (DeiT)**
**Title:** Training data-efficient image transformers & distillation through attention  
**Authors:** Hugo Touvron et al.  
**Link:** [https://arxiv.org/abs/2012.12877](https://arxiv.org/abs/2012.12877)  

---

## **💀 Future Improvements**
🚀 **Integrate more advanced Transformer models** (Swin Transformer, ConvNeXt)  
📈 **Improve UI with Bootstrap or React**  
🎯 **Deploy on cloud for public access**  

---

👨‍💻 **Developed by [Avik Pramanick]**  
🗓️ **Last Updated: 2025**  
🔗 **GitHub Repository:** [https://github.com/Avikg/dog_vs_not_dog_web_app]  

