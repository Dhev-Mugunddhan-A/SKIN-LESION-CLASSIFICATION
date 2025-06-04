# SKIN-LESION-CLASSIFICATION
SKIN CANCER–MELANOMA PREDICTION  AND SKIN LESION CLASSIFICATION USING  DEEP LEARNING MODELS 


## Overview
This project focuses on the categorization of skin lesions from dermoscopic images to aid in efficient and accurate diagnosis for the Computer Aided Diagnostic(CAD) Systems. This project is done as a part of our Final year Capstone Project under the course B.Tech AI and Data Science at Shiv Nadar University Chennai. The project duration spanned across two semesters with Phase-1 in the 7th Sem ( accounting for 3 credits) and Phase -2 in the 8-th  Sem (accounting for 6 credits) 

## PHASE 1
Leveraging the ISIC segmentation and classification challenge datasets, the primary goal is to build a model with fewer parameters and lower runtime complexity during inference while maintaining high diagnostic performance.

### Key Objectives
- Develop a computationally efficient model for skin lesion diagnosis.
- Minimize runtime complexity and the number of parameters in the model.
- Achieve accurate segmentation and classification of skin lesions.

### Dataset
The project utilized the ISIC segmentation and classification challenge 2018 Task 1 and Task3 datasets respectively, which consist of dermoscopic images annotated for skin lesion analysis.
You could find the link to the datasets [here](https://challenge.isic-archive.com/data/#2018). 

### Methodology
The project involves a four-stage pipeline:
1. **Pre-processing**:
   - Image resizing.
   - Contrast enhancement to improve image quality.

2. **Segmentation**:
   - UNet architecture for identifying the Region of Interest (ROI).

3. **Feature Extraction**:
   - **ORB (Oriented FAST and Rotated BRIEF)**: For local feature extraction.
   - **Deep Convolutional Neural Networks (DCNN)**: Base models include ResNet50 and ResNet151 using transfer learning.

4. **Feature Fusion and Classification**:
   - Fusion of DCNN and ORB features using:
     - **Hadamard Product**.
     - **Attention-based fusion techniques**.
   - Classification using the **xgBoost model and ANN**.

### Results and Findings
- The fusion of features using the Hadamard Product demonstrated lower computational expense and better performance compared to alternative approaches.
- The DCNN feature extraction combined with ORB features yielded superior results in terms of accuracy and efficiency.

### Models and Techniques
- **Base Models**: ResNet50 and ResNet151 (transfer learning).
- **Pre-processing**: Image resizing and contrast enhancement.
- **Segmentation**: UNet.
- **Feature Extraction**: ORB and DCNN.
- **Feature Fusion**: Hadamard Product and Attention-based methods.
- **Classifier**: xgBoost, ANN

### Model Performance Summary
| Model                                      | Trainable Parameters | Preprocessing Techniques | Input Image Size | Segmentation | Accuracy (Train/Test) |
|-------------------------------------------|-----------------------|---------------------------|------------------|--------------|------------------------|
| ResNet50                                  | 23M                  | Resize, Contrast          | 224x224          | No           | 90%/50%               |
| ResNet152                                 | 60M                  | Resize, Contrast          | 224x224          | No           | 90%/60%               |
| xgBoost                                   | -                    | Resize, Contrast          | 256x256          | Yes          | NIL/60.25%            |
| DCNN with Feature Fusion (Hadamard)       | 29M                  | Resize, Contrast          | 256x256          | Yes          | 95.66%/57.01%         |
| DCNN with Feature Fusion (Attention Net)  | 31M                  | Resize, Contrast          | 256x256          | Yes          | 66.78%/60%            |

### Advantages of the Approach
- Reduced computational cost during inference.
- Improved accuracy for skin lesion classification.
- Effective fusion of handcrafted ORB features and deep learning-based features.


### How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/Dhev-Mugunddhan-A/skin-lesion-categorization.git
   ```
2. Now run those ipynb files in your local environment, colab or any other cloud engines. Use GPU for faster training and evaluation. 
   a. The file named ISIC segmentation will contain codes for the segmentation task and you could save those segmented images in a separate folder. 
   b. Now run the ISIC-Task-3 notebook to train and test models on the classification task. You could replace the segmented image folder in place of input image folder in the dataloaders. 

3. You could also look into the phase-I-report for a detailed info about our project.


## PHASE 2
This directory contains the Phase II project report and associated code for the skin lesion classification project, conducted as part of the Bachelor of Technology degree in Artificial Intelligence & Data Science at Shiv Nadar University Chennai. The project focuses on classifying skin lesions, particularly distinguishing between benign and malignant cases, using advanced deep learning techniques on the challenging ISIC 2024 dataset.

## Overview

Skin cancer is a prevalent and potentially life-threatening disease where early and accurate diagnosis is critical for effective treatment. Computer-aided diagnosis (CAD) systems utilizing deep learning have shown significant promise in automating this process with high accuracy . This project aimed to develop a robust, accurate, and scalable deep learning model for skin lesion classification . The study addresses key challenges inherent in the ISIC 2024 dataset, such as severe class imbalance, variable image quality, multimodal input (images and patient metadata), and missing data. By leveraging various deep learning architectures, advanced data processing techniques, and ensemble learning, this research contributes to the advancement of AI-assisted dermatology, aiming for enhanced accuracy and reliability in early detection.

## Key Objectives

The primary objectives of this project were:

*   To **develop a robust, accurate, and scalable deep learning model** for skin lesion classification [2].
*   To **utilize the ISIC 2024 dataset**, which provides a comprehensive and large-scale collection of dermoscopic images and patient metadata.
*   To **address significant challenges** posed by the dataset, including severe class imbalance, variable image quality, multimodal input, and missing data.
*   To **explore and evaluate various state-of-the-art deep learning architectures**, including CNN-based models, Vision Transformers, and multimodal architectures.
*   To **implement advanced data processing techniques** such as imputation strategies, feature engineering, and class balancing methods.
*   To **investigate the benefits of multimodal integration** by combining image and metadata features.
*   To **evaluate the performance of different models** and techniques to identify effective strategies for dermatological diagnosis.

## Dataset

The project primarily utilized the **ISIC 2024 Challenge Dataset**. This dataset is a comprehensive, large-scale collection designed for advancing skin lesion analysis using machine learning and computer vision.

Key characteristics and challenges of the dataset include:

*   It comprises **over 400,000 high-resolution RGB dermoscopic images**.
*   It includes a **wide array of skin lesion types**, encompassing both benign and malignant cases, designed to reflect real-world clinical variability.
*   It is a **multimodal dataset**, containing both dermoscopic images and patient metadata.
*   It presents a **huge class imbalance ratio** in the training set, with a disproportionate distribution between benign (Class 0: 400,666 samples) and malignant (Class 1: 393 samples) samples. This extreme rarity of the positive class can significantly bias learning algorithms.
*   The images are often of **variable quality** and in **irregular shapes**.
*   The metadata contains a **lot of null values**.

The task is a **binary classification problem** to distinguish between malignant and benign skin lesions. You could find the link to the dataset [here](https://www.kaggle.com/competitions/isic-2024-challenge)

## Methodologies

The project adopted a comprehensive methodology to tackle the skin lesion classification task and its associated challenges. The overall data pipeline involves separate processing streams for image and metadata before fusion for classification.

Key methodologies employed include:

*   **Data Collection and Preprocessing**:
    *   Handling **missing data** using strategies like Iterative Imputer and KNN Imputer.
    *   **Image Preprocessing** including decoding, resizing, augmentation (random flip, brightness, contrast), and normalization. Images were resized to various dimensions, including 224x224 and 128x128.
    *   **Feature Engineering** on metadata, including age binning, one-hot encoding for categorical features, and feature hashing for high-cardinality features.
    *   **Class Imbalance Handling** using techniques such as class weighting, oversampling (RandomOverSampler, SMOTETomek), and undersampling (RandomUnderSampler). Stratified splitting was used to maintain class distribution and avoid data leakage based on patient IDs.
*   **Image Segmentation**: Used **DeepLabv3+** for lesion segmentation, which helps improve classification accuracy and interpretability by localizing pathological regions.
*   **Feature Selection and Engineering**: Transformed and engineered metadata into a machine-readable format for integration with visual data.
*   **Model Architectures**: Explored various state-of-the-art models:
    *   **CNN-Based Models**: ResNet (ResNet-18, ResNet-50, ResNet-101, ResNet-152, ResNetV2-50), DenseNet (DenseNet-169). Often used as backbones in multimodal models.
    *   **Transformer-Based Methods**: Vision Transformers (ViT), MedMamba, MobileViT. MobileViT and MedMamba architectures were adapted for 2D medical images.
    *   **Multimodal Models**: Architectures that combined image features (from CNN or Transformer backbones) and tabular metadata through concatenation and dense layers.
    *   **Tree-Based Models**: XGBoost, LightGBM, CatBoost. Used on metadata features. Ensemble techniques, including stacking, were explored.
*   **Training and Optimization**:
    *   **Optimizers**: Adam, Nadam, AdamW.
    *   **Loss Functions**: Binary Crossentropy, Squared Hinge, Binary Focal Crossentropy, CrossEntropyLoss (for models using PyTorch). Focal Loss and Binary Focal Crossentropy with class balancing were particularly used to mitigate the class imbalance issue.
    *   **Training Strategies**: Learning rate schedules (Piecewise Constant Decay), Callbacks such as EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint. Class weights were applied during training.
*   **Model Evaluation**: Performance was evaluated using metrics like AUC (Area Under the Curve), particularly pAUC (partial AUC) with ROC and Precision-Recall curves, Accuracy, and LogLoss.

## Results and Findings

The experimentation phase involved evaluating various modeling strategies on both unimodal (image or metadata only) and multimodal data.

Key findings include:

*   Multimodal approaches, integrating both image data and patient metadata, consistently demonstrated **superior performance** compared to image-only or metadata-only models. This highlights the value of combining diverse information sources for improved diagnostic accuracy.
*   The use of **ensemble learning techniques further improved performance**, reducing variance and increasing robustness.
*   Appropriate **data preprocessing techniques**, especially handling class imbalance and missing values, were crucial for building effective models.
*   Refining lesion segmentation was identified as a way to enhance classification accuracy.
*   Selecting **appropriate base architectures** (such as DenseNet or ViT/MedMamba) and carefully optimizing model complexity are key factors in developing effective diagnostic tools.

These findings suggest that integrating multimodal data and employing ensemble techniques are promising directions for improving AI-driven dermatological diagnosis.

## Models and Result Table

The project explored numerous model variations with different preprocessing and training strategies. Below is a summary of reported results for various models, showing their Public and Private test scores (metrics based on the ISIC 2024 challenge, often pAUC or similar competition metrics). Lower scores are generally better in this context (e.g., logloss).

| Competition Submission Name                 | Models Used                                      | Training Parameters (Total Params / Data Size) | Test - Public Score | Test - Private Score | Train AUC | Val AUC | 
| :------------------------------------------ | :----------------------------------------------- | :--------------------------------------------- | :------------------ | :------------------- | :-------- | :------- |
| Tree based models version 9                 | XGBoost, LightGBM, CatBoost                      | 73 features / 10218 samples                    | 0.16540             | 0.14565              | 1         | 0.9978   |
| Dataset EDA – version 6                     | XGBoost Binary Classifier                        | 73 features                                    | 0.16030             | 0.14480              | 1         | 0.9856   |
| Dataset – EDA ver-sion 4                    | XGBoost                                          | 73 features                                    | 0.16416             | 0.13656              | 1         | 0.945    |
| CNN based models version 32                 | DenseNet-169, MLP with Dense Layers              | 15M                                            | 0.15238             | **0.13614**          | 0.9956    | 0.9913   | 
| Basic model building version – 21           | ResNet-101, MLP with Dense Layers                | 50.2M                                          | 0.14285             | 0.12706              | 1         | 1        | 
| Basic model building version 20             | ResNet V2 50, MLP with Dense Layers              | 75.5M                                          | 0.14620             | 0.12413              | 0.9709    | 0.9198   | 
| Basic model version 24                      | ResNet-50, MLP with Dense Layers                 | 439M                                           | 0.14666             | 0.12247              | 0.998     | 0.9485   | 
| ISIC Medmamba version 4                     | MedMamba                                         | 14M                                            | 0.13613             | 0.12280              | 0.9588    | 0.89     | 
| Vision transform-ers version 11             | Vision Transformer, MLP                          | 23M                                            | 0.13676             | 0.11722              | 0.9697    | 0.98     |
| Densenet - model                            | DenseNet-169, MLP with Dense Layers              | 14.94M                                         | 0.11977             | 0.11792              | 0.9885    | 0.9255   |
| Basic model version 27                      | DenseNet-169, MLP with Dense Layers              | 20M                                            | 0.14307             | 0.11759              | 0.9944    | 0.997    |
| Vision transform-ers version 9              | Vision Trans-former, MLP                         | 10M                                            | 0.12120             | 0.11688              | 0.9998    | 0.9995   |
| ISIC Medmamba version 6                     | MedMamba                                         | 16M                                            | 0.13825             | 0.11579              | 0.9678    | 0.8445   |
| Mobile ViT ver-sion 2                       | Mobile ViT, MLP                                  | 4M                                             | 0.13332             | **0.11469**          | 0.9984    | 0.9972   | 
| Tree based models version 10                | XGBoost, LightGBM, CatBoost                      | 73 features / 79386 samples                    | 0.12443             | 0.11055              | 0.9788    | 0.9122   | 
| Basic model- version 4                      | CNN, FCNN(Dense Layers)                          | 26.59M                                         | 0.11640             | 0.10998              | 0.9855    | 0.9898   | 
| Vision transform-ers version 5              | Vision Trans-former, MLP                         | 20M                                            | 0.08228             | 0.09158              | 0.8806    | 0.889    | 
| Mobile ViT ver-sion 5                       | Mobile ViT, MLP                                  | 1.86M                                          | 0.12237             | 0.08652              | 0.9986    | 0.9723   |
| Dataset EDA ver-sion 8                      | XGBoost                                          | 73 features                                    | 0.15839             | 0.13121              | 0.9985    | 0.94     |
| Tree based models version 12                | XGBoost, LightGBM, CatBoost                      | 73 features / 5109 samples                     | 0.15628             | 0.13445              | 1         | 0.9665   |
| Tree based models version 11                | XGBoost, LightGBM, CatBoost (Stacking)           | 73 features / 102180 samples                   | 0.02155             | 0.02104              | 0.9855    | 0.6566   | 
| Basic model- version 15                     | ResNetV2-50, FCNN (Dense Lay-ers)                | 25M                                            | 0.04408             | 0.04972              | 0.9999    | 0.9737   | 
| Basic model - version 9                     | ResNet-18, MLP with Dense Layers                 | 11.7M                                          | 0.03305             | 0.04173              | 0.1681    | 0.6758   | 
| Basic model- version 5                      | ResNet-18, MLP with Dense Layers                 | 12.8M                                          | 0.02920             | 0.03494              | 0.4566    | 0.5455   | 
| Basic model - version 11                    | ResNet-152, MLP with Dense Layers                | 59.5M                                          | 0.02000             | 0.02000              | 0.2277    | 0.5059   | 
| CNN based models version 31                 | DenseNet-169, MLP with Dense Layers              | 20M                                            | 0.00944             | **0.00947**          | 0.9982    | 0.9986   |
| Mobile ViT ver-sion 7                       | Mobile ViT, MLP                                  | 5M                                             | 0.04962             | 0.04613              | 1         | 0.9926   | 
| Vision transform-ers version 6              | Vision Trans-former, MLP                         | 20M                                            | 0.07985             | 0.009617             | 0.9035    | 0.9159   |
| Basic model version 26                      | DenseNet-169, MLP with Dense Layers              | 20M                                            | 0.07654             | 0.05751              | 0.9152    | 0.9943   |
| ISIC Medmamba version 3                     | MedMamba, Tabular FC Layers                      | 14.99M                                         | 0.12188             | 0.10833              | N/A       | N/A      |
| ISIC Medmamba version 1                     | MedMamba (Transfer Learning)                     | 14.45M                                         | 0.09298             | 0.09374              | N/A       | N/A      | 



The **CNN based models version 31** and **Mobile ViT version 2** models achieved the lowest Private scores among the detailed results, suggesting strong performance on the hidden test set data.

## How to Reproduce or Use the Repository

The repository is organized into `PHASE 1` and `PHASE 2` directories, reflecting the project's progression. The `PHASE 2` folder contains the core components related to the deep learning classification work.


To use this repository:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Dhev-Mugunddhan-A/skin-lesion-categorization.git
    ```
2.  **Navigate to the `PHASE 2` directory:**
    ```bash
    cd PHASE2\
    ```
3.  **Install dependencies:** You will need Python and relevant libraries for deep learning (TensorFlow/Keras, PyTorch), data processing (pandas, numpy, scikit-learn), image handling (opencv, PIL), and potentially Streamlit.
4.  **Run the notebooks:** Explore the `.ipynb` files in the `Code files` directory to understand the data preprocessing, model implementations (CNN, ViT, MedMamba, MobileViT, Tree-based), training procedures, and evaluation.
5.  **Run the Streamlit web application:** Navigate to the `streamlit webapp` directory and run the application using Streamlit. It likely provides a user interface to test the trained models.
    ```bash
    cd streamlit\ webapp
    streamlit run app.py 
    ```
    The web application includes saved model files (`best_model_mobilevit_v4.keras`, `fusion_model.pth`) which are used for making predictions.

The `Phase_II project report vfinal.pdf` and `Skin lesion classification methodologies and workflow v 3.pdf` documents provide detailed descriptions of the methods, experiments, and results.
You could also visit the following notebooks in Kaggle to start working with directly without any setup. 
* CNN Based models -> https://www.kaggle.com/code/dhevmugunddhana333/cnn-based-models
* Tree based models -> https://www.kaggle.com/code/dhevmugunddhana333/tree-based-models
* medmamba models -> https://www.kaggle.com/code/dhevmugunddhana333/isic-medmamba-submissions
* Vision Transformer models -> https://www.kaggle.com/code/dhevmugunddhana333/vision-transformer-with-keras-3
* MobileViT -> https://www.kaggle.com/code/dhevmugunddhana333/mobilevit-with-keras-3

## Conclusion
This project demonstrates the feasibility of developing an efficient and accurate skin lesion diagnosis system. The proposed methodologies ensure reduced runtime complexity and enhanced performance, making it suitable for practical clinical applications.

## Acknowledgments
- ISIC challenge for providing the datasets.
- Open-source libraries and frameworks that facilitated model development.
- Kaggle platform in which our models were trained, tested and published.

## License
This project is licensed under the MIT License.

## Contact
For further queries or collaborations, feel free to reach out at mugunddhan3@gmail.com.

