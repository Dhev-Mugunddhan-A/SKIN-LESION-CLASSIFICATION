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

model_path = "best_model_mobilevit_v4.keras"
model = keras.models.load_model(model_path)

print("Model loaded successfully")


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

        cat_feature_names = self.preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(
            categorical_cols)
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
        return df.drop(columns=['patient_id', 'image_type', 'tbp_tile_type', 'attribution', 'copyright_license'],
                       errors="ignore")

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




# Load the preprocessor
with open("fitted_preprocessor_v2.pkl", "rb") as f:
    data_object = pickle.load(f)

print("Data preprocessor loaded successfully!")

csv_str = """isic_id,patient_id,age_approx,sex,anatom_site_general,clin_size_long_diam_mm,image_type,tbp_tile_type,tbp_lv_A,tbp_lv_Aext,tbp_lv_B,tbp_lv_Bext,tbp_lv_C,tbp_lv_Cext,tbp_lv_H,tbp_lv_Hext,tbp_lv_L,tbp_lv_Lext,tbp_lv_areaMM2,tbp_lv_area_perim_ratio,tbp_lv_color_std_mean,tbp_lv_deltaA,tbp_lv_deltaB,tbp_lv_deltaL,tbp_lv_deltaLB,tbp_lv_deltaLBnorm,tbp_lv_eccentricity,tbp_lv_location,tbp_lv_location_simple,tbp_lv_minorAxisMM,tbp_lv_nevi_confidence,tbp_lv_norm_border,tbp_lv_norm_color,tbp_lv_perimeterMM,tbp_lv_radial_color_std_max,tbp_lv_stdL,tbp_lv_stdLExt,tbp_lv_symm_2axis,tbp_lv_symm_2axis_angle,tbp_lv_x,tbp_lv_y,tbp_lv_z,attribution,copyright_license
ISIC_0015740,IP_7142616,65.0,male,posterior torso,3.16,TBP tile: close-up,3D: XP,24.25384,19.93738,30.46368,28.38424,38.9395,34.68666,51.47473,54.91541,35.81945,41.35864,3.39651,19.4644,0.2512358,4.316465,2.079433,-5.539191,6.041092,5.446997,0.8947765,Torso Back Top Third,Torso Back,1.520786,8.052259e-13,3.968912,0.7217392,8.130868,0.2307418,1.080308,2.705857,0.3660714,110,-84.29282,1303.978,-28.57605,FNQH Cairns,CC-BY"""

# Convert CSV string to DataFrame
metadata_df = pd.read_csv(StringIO(csv_str))
print("Raw metadata sample:")
print(metadata_df)

# Preprocess Metadata
# Preprocess metadata using the fitted preprocessor
processed_metadata = data_object.transform(metadata_df)

# Remove 'isic_id' as it is not a model input
metadata_input = processed_metadata.drop(columns=['isic_id']).values.astype('float32')

# Reshape to maintain batch dimension
metadata_input = metadata_input.reshape(1, -1)

print("Preprocessed Metadata Sample:")
print(metadata_input)

# Image Processing
def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at path: {image_path}")
    img = cv2.resize(img, (128, 128))  # Resize to model input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Load Image
image_path = "ISIC_0015740.png"
image_input = process_image(image_path)

print("Image shape after processing:", image_input.shape)

# Prediction
prediction = model.predict([image_input, metadata_input])

# Display Result
result = "Malignant" if prediction[0][0] > 0.5 else "Benign"
print(f"Prediction Result: {result}")

