import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import cv2
from io import StringIO
import pickle
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import time
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Convert your local logo to base64
import base64
with open("snu.jpg", "rb") as image_file:
    logo_base64 = base64.b64encode(image_file.read()).decode()

# Add custom styling and header
st.markdown(f"""
    <style>
        /* Full-width blue header with white text */
        .custom-header {{
            width: 100%;
            background-color: #002D72;
            color: white;
            padding: 20px 30px;
            font-size: 26px;
            font-weight: bold;
            position: relative;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            margin-bottom: 25px;
        }}

        /* Top-right logo inside header */
        .custom-header img {{
            position: absolute;
            top: 12px;
            right: 30px;
            height: 50px;
        }}

        /* White background for the body */
        .main, .block-container {{
            background-color: white !important;
            color: black !important;
        }}

        /* Set all text (excluding header) to black */
        h1, h2, h3, h4, h5, h6, p, label, span, div {{
            color: black !important;
        }}

        /* Keep custom header text white */
        .custom-header, .custom-header * {{
            color: white !important;
        }}

        /* Style all buttons to black with white text */
        .stButton > button {{
            background-color: #000000 !important;
            color: white !important;
            border: none;
            padding: 0.6em 1.2em;
            font-size: 16px;
            font-weight: 600;
            border-radius: 8px;
            transition: background-color 0.3s ease;
        }}

        .stButton > button:hover {{
            background-color: #333333 !important;
            color: white !important;
        }}

        /* Hide Streamlit menu and footer */
        #MainMenu, footer {{
            visibility: hidden;
        }}
    </style>

    <!-- Header with logo -->
    <div class="custom-header">
        Malignancy Prediction App
        <img src="data:image/jpg;base64,{logo_base64}">
    </div>
""", unsafe_allow_html=True)


class DataPreprocessor:
    def __init__(self):
        self.preprocessor = None
        self.train_columns = None

    def fit_transform(self, df):
        """Preprocess training data and store transformations."""
        df = df.copy()
        df = self._drop_irrelevant_columns(df)
        df = self._drop_train_only_columns(df)
        categorical_cols, numerical_cols = self._identify_column_types(df)
        
        # Define preprocessing pipelines
        numerical_pipeline = Pipeline([
            ("imputer", KNNImputer()),
            ("scaler", StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        
        self.preprocessor = ColumnTransformer([
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ])
        
        transformed_data = self.preprocessor.fit_transform(df)
        
        cat_feature_names = self.preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_cols)
        all_columns = numerical_cols + list(cat_feature_names)
        
        df_processed = pd.DataFrame(transformed_data, columns=all_columns)
        df_processed["isic_id"] = df["isic_id"].values
        df_processed["target"] = df["target"].values
        
        self.train_columns = df_processed.columns  # Store train columns
        return df_processed

    def transform(self, df):
        """Preprocess test data using stored transformations from training."""
        df = df.copy()
        df = self._drop_irrelevant_columns(df)
        
        transformed_data = self.preprocessor.transform(df)
        cat_feature_names = self.preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out()
        all_columns = self.train_columns[:-2]  # Exclude 'isic_id' and 'target'
        
        df_processed = pd.DataFrame(transformed_data, columns=all_columns)
        df_processed["isic_id"] = df["isic_id"].values
        # Align test dataset with train columns
        df_processed = self._align_train_test_columns(df_processed)
        return df_processed

    def _drop_irrelevant_columns(self, df):
        """Remove unnecessary columns."""
        return df.drop(columns=['patient_id','image_type', 'tbp_tile_type', 'attribution', 'copyright_license'], errors="ignore")

    def _drop_train_only_columns(self, df):
        """Remove columns that are present only in the train set and not in the test set."""
        drop_train_only_columns = [
            'lesion_id', 'iddx_full', 'iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5',
            'mel_mitotic_index', 'mel_thick_mm', 'tbp_lv_dnn_lesion_confidence'
        ]
        return df.drop(columns=drop_train_only_columns, errors='ignore')
        
    def _identify_column_types(self, df):
        """Identify categorical and numerical columns, excluding 'isic_id' and 'target'."""
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col != "isic_id"]
        numerical_cols = [col for col in numerical_cols if col != "target"]
        return categorical_cols, numerical_cols
    
    def _align_train_test_columns(self, df):
        """Ensure test data has the same columns as train data."""
        train_cols = list(self.train_columns)
        train_cols.remove('target')
        missing_cols = set(train_cols) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        return df[train_cols]




#############################################
# Existing MobileViT & Preprocessor Code
#############################################
# Load the Keras MobileViT model
model_path = "best_model_mobilevit_v4.keras"
mobilevit_model = keras.models.load_model(model_path,compile=False)
st.success("✅ MobileViT model loaded successfully!")

# Load the fitted preprocessor for metadata
with open("fitted_preprocessor_v2.pkl", "rb") as f:
    data_object = pickle.load(f)
st.success("✅ Data preprocessor loaded successfully!")

st.title("🔬 Malignancy Prediction App")
st.write("Upload metadata & an image to predict if a lesion is benign or malignant.")

# --- CSV Metadata Input ---
st.subheader("📑 Enter Metadata CSV")
csv_input = st.text_area(
    "Paste your CSV string here:",
    # value="""isic_id,patient_id,age_approx,sex,anatom_site_general,clin_size_long_diam_mm,image_type,tbp_tile_type,tbp_lv_A,tbp_lv_Aext,tbp_lv_B,tbp_lv_Bext,tbp_lv_C,tbp_lv_Cext,tbp_lv_H,tbp_lv_Hext,tbp_lv_L,tbp_lv_Lext,tbp_lv_areaMM2,tbp_lv_area_perim_ratio,tbp_lv_color_std_mean,tbp_lv_deltaA,tbp_lv_deltaB,tbp_lv_deltaL,tbp_lv_deltaLB,tbp_lv_deltaLBnorm,tbp_lv_eccentricity,tbp_lv_location,tbp_lv_location_simple,tbp_lv_minorAxisMM,tbp_lv_nevi_confidence,tbp_lv_norm_border,tbp_lv_norm_color,tbp_lv_perimeterMM,tbp_lv_radial_color_std_max,tbp_lv_stdL,tbp_lv_stdLExt,tbp_lv_symm_2axis,tbp_lv_symm_2axis_angle,tbp_lv_x,tbp_lv_y,tbp_lv_z,attribution,copyright_license
# ISIC_0015740,IP_7142616,65.0,male,posterior torso,3.16,TBP tile: close-up,3D: XP,24.25384,19.93738,30.46368,28.38424,38.9395,34.68666,51.47473,54.91541,35.81945,41.35864,3.39651,19.4644,0.2512358,4.316465,2.079433,-5.539191,6.041092,5.446997,0.8947765,Torso Back Top Third,Torso Back,1.520786,8.052259e-13,3.968912,0.7217392,8.130868,0.2307418,1.080308,2.705857,0.3660714,110,-84.29282,1303.978,-28.57605,FNQH Cairns,CC-BY"""
    value='''isic_id,patient_id,age_approx,sex,anatom_site_general,clin_size_long_diam_mm,image_type,tbp_tile_type,tbp_lv_A,tbp_lv_Aext,tbp_lv_B,tbp_lv_Bext,tbp_lv_C,tbp_lv_Cext,tbp_lv_H,tbp_lv_Hext,tbp_lv_L,tbp_lv_Lext,tbp_lv_areaMM2,tbp_lv_area_perim_ratio,tbp_lv_color_std_mean,tbp_lv_deltaA,tbp_lv_deltaB,tbp_lv_deltaL,tbp_lv_deltaLB,tbp_lv_deltaLBnorm,tbp_lv_eccentricity,tbp_lv_location,tbp_lv_location_simple,tbp_lv_minorAxisMM,tbp_lv_nevi_confidence,tbp_lv_norm_border,tbp_lv_norm_color,tbp_lv_perimeterMM,tbp_lv_radial_color_std_max,tbp_lv_stdL,tbp_lv_stdLExt,tbp_lv_symm_2axis,tbp_lv_symm_2axis_angle,tbp_lv_x,tbp_lv_y,tbp_lv_z,attribution,copyright_license
ISIC_7031844,IP_1959239,60.0,male,lower extremity,4.78,TBP tile: close-up,3D: XP,30.5357014648896,23.6495223175003,21.7813108726253,25.160111128898,37.5080600309161,34.5301476664152,35.5005057689911,46.7726571614438,37.9219983466586,45.6893030182267,7.48733345843498,31.4289164265365,1.70035725760121,6.88617914738937,-3.37880025627274,-7.7673046715681,8.76691223804216,7.15284922303556,0.748028862591454,Right Leg,Right Leg,3.32240561916404,5.21206402481766e-06,8.43787206140855,4.37460988678988,15.3401035694927,1.2278434619367,2.54167858989563,3.62715945239861,0.681063122923588,150,206.784057617188,599.679565429688,4.529296875,Frazer Institute Dermatology Research Centre,CC-BY'''

)

if csv_input:
    try:
        metadata_df = pd.read_csv(StringIO(csv_input))
        st.write("Raw metadata sample:")
        st.dataframe(metadata_df.style.set_properties(**{'background-color': '#1e1e2f', 'color': 'white'}))
        
        # Process metadata
        processed_metadata = data_object.transform(metadata_df)
        metadata_input = processed_metadata.drop(columns=['isic_id']).values.astype('float32').reshape(1, -1)
    except Exception as e:
        st.error(f"Error processing metadata: {e}")

# --- Image Upload for MobileViT & Fusion Model ---
st.subheader("📷 Upload Image")
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

def process_image_mobilevit(image_file):
    # For MobileViT: process image to the expected input size
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Uploaded file is not a valid image.")
    img = cv2.resize(img, (128, 128))  # Resize to model's expected input
    img = img.astype("float32") / 255.0  
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

#############################################
# Additional Fusion Model (PyTorch) Code
#############################################
# Define FusionModel architecture (must match training)
class FusionModel(nn.Module):
    def __init__(self, orb_input_dim=3200, num_classes=7):
        super(FusionModel, self).__init__()
        self.image_model = models.resnet18(pretrained=False)
        self.image_model.fc = nn.Identity()  # outputs 512-dim feature vector
        self.orb_fc = nn.Linear(orb_input_dim, 512)
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, image, orb_features):
        img_feat = self.image_model(image)
        orb_feat = self.orb_fc(orb_features)
        fused = img_feat * orb_feat  # Hadamard product
        out = self.classifier(fused)
        return out

# Load Fusion Model weights
fusion_model = FusionModel()
fusion_model.load_state_dict(torch.load("fusion_model.pth", map_location=torch.device('cpu')))
fusion_model.eval()

# Preprocessing transform for the image branch (Fusion Model)
fusion_preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Helper functions for ORB processing (Fusion Model)
def enhance_contrast_resize(image, contrast_factor=1.5, size=(256, 256)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=contrast_factor, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    resized_image = cv2.resize(enhanced_image, size)
    return resized_image

def extract_orb_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    return keypoints, descriptors

#############################################
# Model Selection & Prediction Section
#############################################
st.subheader("🔍 Prediction")
# Let the user choose which model to use
prediction_model_choice = st.radio("Select Prediction Model", options=["MobileViT", "Fusion Model"], index=0)

if uploaded_file is not None:
    # For MobileViT, process image once (we can reuse for fusion model if needed)
    try:
        # Reset file pointer for each use
        uploaded_file.seek(0)
        image_input_mobilevit = process_image_mobilevit(uploaded_file)
        st.write("MobileViT image processed successfully!")
    except Exception as e:
        st.error(f"Error processing image for MobileViT: {e}")

    # Also, prepare image for Fusion Model prediction
    try:
        uploaded_file.seek(0)
        input_image = Image.open(uploaded_file).convert("RGB")
        # Preprocess image branch for Fusion Model
        image_tensor = fusion_preprocess_transform(input_image).unsqueeze(0)
        
        # Process for ORB branch: convert PIL image to BGR for OpenCV
        image_np = np.array(input_image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        processed_for_orb = enhance_contrast_resize(image_bgr, contrast_factor=1.5, size=(256,256))
        keypoints, descriptors = extract_orb_features(processed_for_orb)
        
        # Display image with ORB keypoints (Fusion Model visualization)
        image_with_keypoints = cv2.drawKeypoints(processed_for_orb, keypoints, None, color=(255, 0, 0), flags=4)
        st.image(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB), caption="ORB Keypoints", use_column_width=True)
        
        # Process ORB descriptors: take first 100 descriptors and pad to 3200 values if needed.
        if descriptors is not None:
            descriptors = descriptors.astype(np.float32)
            descriptors = descriptors[:100]
            flat_descriptors = descriptors.flatten()
            if len(flat_descriptors) < 3200:
                flat_descriptors = np.pad(flat_descriptors, (0, 3200 - len(flat_descriptors)), mode='constant', constant_values=0)
        else:
            flat_descriptors = np.zeros(3200, dtype=np.float32)
        orb_tensor = torch.tensor(flat_descriptors).unsqueeze(0)
    except Exception as e:
        st.error(f"Error processing image for Fusion Model: {e}")

if st.button("🚀 Predict Malignancy"):
    if csv_input and uploaded_file is not None:
        progress_bar = st.progress(0)
        status_text = st.empty()
        for percent_complete in range(100):
            time.sleep(0.03)
            progress_bar.progress(percent_complete + 1)
            status_text.text(f"🔄 Processing... {percent_complete + 1}%")
            
        if prediction_model_choice == "MobileViT":
            # MobileViT prediction using Keras model
            prediction = mobilevit_model.predict([image_input_mobilevit, metadata_input])
            if(prediction[0][0]<=0.5):
                result='Malignant'
            else:
                result = 'Benign'

            st.success(f"✅ MobileViT Prediction Complete: {result}")
        else:
            # Fusion Model prediction using PyTorch
            with torch.no_grad():
                output = fusion_model(image_tensor, orb_tensor)
                prediction_tensor = torch.argmax(output, dim=1)
            class_labels = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
            st.success(f"✅ Fusion Model Prediction Complete: {class_labels[prediction_tensor.item()]}")
    else:
        st.warning("⚠️ Please provide both metadata and an image.")

st.markdown("""
    <style>
        .custom-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 50px;
            font-size: 16px;
            font-family: 'Segoe UI', sans-serif;
        }

        .custom-table th, .custom-table td {
            border: 1px solid #ccc;
            padding: 10px 15px;
            text-align: left;
        }

        .custom-table th {
            background-color: #002D72; /* SNU dark blue */
            color: white;
        }

        .custom-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }

        .custom-table tr:nth-child(odd) {
            background-color: #ffffff;
        }
    </style>

    <table class="custom-table">
        <thead>
            <tr>
                <th>Abbreviation</th>
                <th>Full Name</th>
            </tr>
        </thead>
        <tbody>
            <tr><td><strong>MEL</strong></td><td>Melanoma</td></tr>
            <tr><td><strong>NV</strong></td><td>Melanocytic nevus</td></tr>
            <tr><td><strong>BCC</strong></td><td>Basal cell carcinoma</td></tr>
            <tr><td><strong>AKIEC</strong></td><td>Actinic keratosis / Bowen’s disease</td></tr>
            <tr><td><strong>BKL</strong></td><td>Benign keratosis</td></tr>
            <tr><td><strong>DF</strong></td><td>Dermatofibroma</td></tr>
            <tr><td><strong>VASC</strong></td><td>Vascular lesion</td></tr>
        </tbody>
    </table>
""", unsafe_allow_html=True)
