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

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration classes (assuming these were in your original code)
class BMDConfig:
    model_path = "resnet50_birth_death_marriage.pth"
    class_names = ["Birth", "Death", "Marriage"]
    num_classes = len(class_names)

class LayoutConfig:
    model_path = "best_document_classifier.pth"
    class_names = ["Table", "List", "Form"]
    num_classes = len(class_names)

# Preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Placeholder for your custom DocumentClassifier (replace with actual definition if different)
class DocumentClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DocumentClassifier, self).__init__()
        self.resnet = models.resnet50(weights=None)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# Model loading functions
def load_bmd_model():
    try:
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, BMDConfig.num_classes)
        )
        model.load_state_dict(torch.load(BMDConfig.model_path, map_location=DEVICE))
        model.eval()
        model.to(DEVICE)
        return model
    except Exception as e:
        st.error(f"Error loading BMD model: {e}")
        return None

def load_layout_model():
    try:
        model = DocumentClassifier(num_classes=LayoutConfig.num_classes)
        checkpoint = torch.load(LayoutConfig.model_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        model.to(DEVICE)
        return model
    except Exception as e:
        st.error(f"Error loading Layout model: {e}")
        return None

def load_models():
    bmd_model = load_bmd_model()
    layout_model = load_layout_model()
    return bmd_model, layout_model

# Prediction functions
def predict_bmd(model, image):
    try:
        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
        return predicted.item(), probs[0].cpu().numpy()
    except Exception as e:
        st.error(f"Error predicting BMD: {e}")
        return None, None

def predict_layout(model, image):
    try:
        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            confidence = float(probs[0][predicted].item() * 100)
        return predicted.item(), confidence, probs[0].cpu().numpy()
    except Exception as e:
        st.error(f"Error predicting Layout: {e}")
        return None, None, None

# Process zip file contents
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
    image_files = []
    for root, _, files in os.walk(temp_extract_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        return bmd_results, layout_results, "No valid images found in zip file."
    
    bmd_results["total"] = len(image_files)
    layout_results["total"] = len(image_files)
    
    # Create output subdirectories
    for bmd_class in BMDConfig.class_names:
        bmd_path = os.path.join(output_base, bmd_class)
        os.makedirs(bmd_path, exist_ok=True)
        for layout_class in LayoutConfig.class_names + ['uncertain']:
            os.makedirs(os.path.join(bmd_path, layout_class), exist_ok=True)
    
    # Process each image
    for i, image_path in enumerate(image_files):
        progress_bar.progress((i + 1) / len(image_files))
        status_text.text(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            if bmd_model:
                bmd_class, bmd_probs = predict_bmd(bmd_model, image)
                if bmd_class is not None:
                    bmd_class_name = BMDConfig.class_names[bmd_class]
                    bmd_confidence = float(bmd_probs[bmd_class] * 100)
                    bmd_results["processed"] += 1
                    bmd_results["class_counts"][bmd_class_name] += 1
                    bmd_results["file_results"].append({
                        'file': os.path.basename(image_path), 
                        'prediction': bmd_class_name,
                        'confidence': bmd_confidence
                    })
                    
                    if layout_model:
                        layout_class, layout_confidence, all_probs = predict_layout(layout_model, image)
                        if layout_class is not None:
                            layout_class_name = LayoutConfig.class_names[layout_class]
                            layout_file_result = {
                                "file": os.path.basename(image_path),
                                "bmd_class": bmd_class_name
                            }
                            
                            target_folder = os.path.join(output_base, bmd_class_name)
                            if layout_confidence >= confidence_threshold:
                                dest_path = os.path.join(target_folder, layout_class_name, os.path.basename(image_path))
                                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                                shutil.copy(image_path, dest_path)
                                layout_results["processed"] += 1
                                layout_results["class_counts"][layout_class_name] += 1
                                layout_file_result.update({
                                    "status": "classified",
                                    "layout_class": layout_class_name,
                                    "confidence": f"{layout_confidence:.2f}%"
                                })
                            else:
                                dest_path = os.path.join(target_folder, "uncertain", os.path.basename(image_path))
                                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                                shutil.copy(image_path, dest_path)
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
                'file': os.path.basename(image_path), 
                'error': str(e)
            })
    
    # Clean up extracted temp directory
    shutil.rmtree(temp_extract_dir)
    
    return bmd_results, layout_results, f"Processed {len(image_files)} files from zip."

# Main app
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
        st.subheader("Upload Single Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if st.button("Classify Image"):
                if use_bmd and bmd_model:
                    bmd_class, bmd_probs = predict_bmd(bmd_model, image)
                    if bmd_class is not None:
                        st.write(f"BMD Prediction: {BMDConfig.class_names[bmd_class]}")
                        st.write(f"Confidence: {bmd_probs[bmd_class]*100:.2f}%")
                
                if use_layout and layout_model:
                    layout_class, layout_confidence, all_probs = predict_layout(layout_model, image)
                    if layout_class is not None:
                        st.write(f"Layout Prediction: {LayoutConfig.class_names[layout_class]}")
                        st.write(f"Confidence: {layout_confidence:.2f}%")
    
    with tab2:
        st.subheader("Upload Zip File of Images")
        zip_file = st.file_uploader("Upload a zip file containing images", type=["zip"])
        
        # Use a temporary directory for output
        output_folder = tempfile.mkdtemp()
        st.write(f"Output will be processed in: {output_folder}")
        
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
                
                # Provide download option for processed files
                for bmd_class in BMDConfig.class_names:
                    for layout_class in LayoutConfig.class_names + ['uncertain']:
                        folder_path = os.path.join(output_folder, "Classified_Documents", bmd_class, layout_class)
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
        st.write("This app classifies documents into Birth/Death/Marriage and Table/List/Form categories using deep learning models.")

if __name__ == "__main__":
    main()
