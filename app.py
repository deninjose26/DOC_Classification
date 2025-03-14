import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import shutil
import zipfile
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import tempfile

# Page configuration
st.set_page_config(
    page_title="Document Type Classifier",
    page_icon="📄",
    layout="wide"
)

# Application title and description
st.title("📄 Document Type Classifier")
st.markdown("Upload images of documents or process a zip file to classify them by document type and layout.")

# Model configurations
class BMDConfig:
    num_classes = 3
    model_path = "resnet50_birth_death_marriage.pth"
    class_names = ['Birth', 'Death', 'Marriage']
    supported_extensions = ['.jpg', '.jpeg', '.png']

class LayoutConfig:
    num_classes = 3
    model_path = "best_document_classifier.pth"
    class_names = ['table', 'list', 'form']
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BMD Classifier model
def load_bmd_model():
    try:
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, BMDConfig.num_classes)
        )
        
        if os.path.exists(BMDConfig.model_path):
            model.load_state_dict(torch.load(BMDConfig.model_path, map_location=DEVICE))
            model.eval()
            model.to(DEVICE)
            return model
        else:
            st.warning(f"BMD Model file '{BMDConfig.model_path}' not found!")
            return None
    except Exception as e:
        st.error(f"Error loading BMD model: {e}")
        return None

# Layout Classifier model
class DocumentClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DocumentClassifier, self).__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def load_layout_model():
    try:
        model = DocumentClassifier(num_classes=LayoutConfig.num_classes)
        if os.path.exists(LayoutConfig.model_path):
            checkpoint = torch.load(LayoutConfig.model_path, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            model.to(DEVICE)
            return model
        else:
            st.warning(f"Layout Model file '{LayoutConfig.model_path}' not found!")
            return None
    except Exception as e:
        st.error(f"Error loading Layout model: {e}")
        return None

# Cache resources
@st.cache_resource
def load_models():
    return load_bmd_model(), load_layout_model()

# Image preprocessing
def preprocess_bmd_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def preprocess_layout_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# Prediction functions
def predict_bmd(model, image):
    if model is None:
        return None, None
    processed_image = preprocess_bmd_image(image)
    with torch.no_grad():
        outputs = model(processed_image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    return predicted_class, probabilities.cpu().numpy()

def predict_layout(model, image):
    if model is None:
        return None, None, None
    processed_image = preprocess_layout_image(image)
    with torch.no_grad():
        output = model(processed_image)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item() * 100
        all_probs = {LayoutConfig.class_names[i]: probabilities[i].item() * 100 
                     for i in range(len(LayoutConfig.class_names))}
    return predicted_class, confidence, all_probs

# Zip file processing
def process_zip(bmd_model, layout_model, zip_file, output_folder, confidence_threshold=50.0):
    output_base = os.path.join(output_folder, "Classified_Documents")
    os.makedirs(output_base, exist_ok=True)
    
    bmd_results = {"total": 0, "processed": 0, 
                   "class_counts": {name: 0 for name in BMDConfig.class_names}, 
                   "file_results": []}
    
    layout_results = {"total": 0, "processed": 0, "skipped": 0, 
                      "class_counts": {name: 0 for name in LayoutConfig.class_names}, 
                      "uncertain": 0, "file_results": []}
    
    if not zip_file:
        return bmd_results, layout_results, "No zip file uploaded."
    
    # Extract zip to a temp directory
    temp_extract_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)
    
    # Gather all image files from the extracted contents
    image_files = [os.path.join(root, f) for root, _, files in os.walk(temp_extract_dir) 
                   for f in files if f.lower().endswith(tuple(BMDConfig.supported_extensions))]
    
    if not image_files:
        shutil.rmtree(temp_extract_dir)
        return bmd_results, layout_results, "No supported image files found in zip."
    
    bmd_results["total"] = len(image_files)
    layout_results["total"] = len(image_files)
    
    for bmd_class in BMDConfig.class_names:
        bmd_path = os.path.join(output_base, bmd_class)
        os.makedirs(bmd_path, exist_ok=True)
        for layout_class in LayoutConfig.class_names + ['uncertain']:
            os.makedirs(os.path.join(bmd_path, layout_class), exist_ok=True)
    
    for i, img_path in enumerate(image_files):
        progress_bar.progress((i + 1) / len(image_files))
        status_text.text(f"Processing {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if bmd_model:
                bmd_class, bmd_probs = predict_bmd(bmd_model, image)
                if bmd_class is not None:
                    bmd_class_name = BMDConfig.class_names[bmd_class]
                    bmd_confidence = float(bmd_probs[bmd_class] * 100)
                    bmd_results["processed"] += 1
                    bmd_results["class_counts"][bmd_class_name] += 1
                    bmd_results["file_results"].append({
                        'file': img_path, 
                        'prediction': bmd_class_name,
                        'confidence': bmd_confidence
                    })
                    
                    if layout_model:
                        layout_class, layout_confidence, all_probs = predict_layout(layout_model, image)
                        if layout_class is not None:
                            layout_class_name = LayoutConfig.class_names[layout_class]
                            layout_file_result = {
                                "file": img_path,
                                "bmd_class": bmd_class_name
                            }
                            
                            target_folder = os.path.join(output_base, bmd_class_name)
                            if layout_confidence >= confidence_threshold:
                                dest_path = os.path.join(target_folder, layout_class_name, os.path.basename(img_path))
                                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                                shutil.copy2(img_path, dest_path)
                                layout_results["processed"] += 1
                                layout_results["class_counts"][layout_class_name] += 1
                                layout_file_result.update({
                                    "status": "classified",
                                    "layout_class": layout_class_name,
                                    "confidence": f"{layout_confidence:.2f}%"
                                })
                            else:
                                dest_path = os.path.join(target_folder, "uncertain", os.path.basename(img_path))
                                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                                shutil.copy2(img_path, dest_path)
                                layout_results["uncertain"] += 1
                                layout_file_result.update({
                                    "status": "uncertain",
                                    "confidence": f"{layout_confidence:.2f}%",
                                    "best_guess": layout_class_name
                                })
                            layout_results["file_results"].append(layout_file_result)
            
        except Exception as e:
            layout_results["skipped"] += 1
            layout_results["file_results"].append({
                'file': img_path, 
                'error': str(e)
            })
    
    # Clean up extracted temp directory
    shutil.rmtree(temp_extract_dir)
    
    return bmd_results, layout_results, f"Processed {len(image_files)} files from zip."

# Main application
def main():
    bmd_model, layout_model = load_models()
    
    tab1, tab2, tab3 = st.tabs(["Upload Single Image", "Process Zip File", "About"])
    
    with st.sidebar:
        st.subheader("Model Settings")
        use_bmd = st.checkbox("Enable Birth/Death/Marriage Classification", value=True)
        use_layout = st.checkbox("Enable Table/List/Form Classification", value=True)
        if not (use_bmd or use_layout):
            st.warning("Please enable at least one classification model")
    
    with tab1:
        uploaded_file = st.file_uploader("Upload a document image", type=["jpg", "jpeg", "png", "bmp"])
        
        image = None
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
        
        if image:
            st.image(image, caption="Document Image", use_column_width=True)
            col1, col2 = st.columns(2)
            
            if use_bmd and bmd_model:
                with col1:
                    st.subheader("Document Type Classification")
                    with st.spinner("Classifying..."):
                        bmd_class, bmd_probs = predict_bmd(bmd_model, image)
                    if bmd_class is not None:
                        st.success(f"Prediction: {BMDConfig.class_names[bmd_class]} Certificate")
                        for j, name in enumerate(BMDConfig.class_names):
                            st.text(f"{name}: {bmd_probs[j] * 100:.2f}%")
            
            if use_layout and layout_model:
                with col2:
                    st.subheader("Layout Classification")
                    with st.spinner("Classifying..."):
                        layout_class, confidence, all_probs = predict_layout(layout_model, image)
                    if layout_class is not None:
                        st.success(f"Prediction: {LayoutConfig.class_names[layout_class]}")
                        st.metric("Confidence", f"{confidence:.2f}%")
                        fig, ax = plt.subplots()
                        ax.barh(list(all_probs.keys()), list(all_probs.values()))
                        st.pyplot(fig)

    with tab2:
        st.subheader("Upload Zip File of Documents")
        zip_file = st.file_uploader("Upload a zip file containing images", type=["zip"])
        
        # Use a temporary directory for output
        output_folder = tempfile.mkdtemp()
        st.write(f"Output will be processed in a temporary folder: {output_folder}")
        
        confidence_threshold = st.slider("Confidence Threshold for Layout (%)", 0.0, 100.0, 50.0)
        process_button = st.button("Process Zip", disabled=not zip_file)
        
        if process_button:
            global progress_bar, status_text
            progress_bar = st.progress(0)
            status_text = st.empty()
            bmd_results, layout_results, message = process_zip(
                bmd_model if use_bmd else None,
                layout_model if use_layout else None,
                zip_file,
                output_folder,
                confidence_threshold
            )
            st.success(message)
            
            if use_bmd and bmd_model and bmd_results["file_results"]:
                st.subheader("BMD Classification Summary")
                st.write(f"Total Documents Processed: {bmd_results['processed']}/{bmd_results['total']}")
                st.write("Distribution:")
                for class_name, count in bmd_results["class_counts"].items():
                    st.write(f"- {class_name}: {count} documents")
                fig, ax = plt.subplots()
                ax.pie(bmd_results["class_counts"].values(), 
                       labels=bmd_results["class_counts"].keys(), 
                       autopct='%1.1f%%')
                ax.set_title("BMD Distribution")
                st.pyplot(fig)
                
            if use_layout and layout_model and layout_results["file_results"]:
                st.subheader("Layout Classification Summary")
                st.write(f"Total Documents Processed: {layout_results['processed']}/{layout_results['total']}")
                st.write(f"Uncertain Classifications: {layout_results['uncertain']}")
                st.write("Distribution of Confident Classifications:")
                for class_name, count in layout_results["class_counts"].items():
                    st.write(f"- {class_name}: {count} documents")
                layout_counts = layout_results["class_counts"].copy()
                layout_counts["uncertain"] = layout_results["uncertain"]
                fig, ax = plt.subplots()
                ax.pie(layout_counts.values(), 
                       labels=layout_counts.keys(), 
                       autopct='%1.1f%%')
                ax.set_title("Layout Distribution")
                st.pyplot(fig)
                
                # Provide download options for processed files
                st.subheader("Download Classified Files")
                for bmd_class in BMDConfig.class_names:
                    for layout_class in LayoutConfig.class_names + ['uncertain']:
                        folder_path = os.path.join(output_base, bmd_class, layout_class)
                        if os.path.exists(folder_path):
                            for file_name in os.listdir(folder_path):
                                file_path = os.path.join(folder_path, file_name)
                                with open(file_path, "rb") as f:
                                    st.download_button(
                                        label=f"Download {bmd_class}/{layout_class}/{file_name}",
                                        data=f,
                                        file_name=file_name,
                                        mime="image/jpeg"
                                    )

    with tab3:
        st.subheader("About")
        st.markdown("This app classifies documents using two models: BMD (ResNet50) and Layout (EfficientNet-B0).")
        st.markdown("Documents are first classified as Birth/Death/Marriage, then within each category as table/list/form.")

if __name__ == "__main__":
    main()
