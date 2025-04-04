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

## PHASE 2
## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/skin-lesion-categorization.git
   ```
2. Now run those ipynb files in your local environment, colab or any other cloud engines. Use GPU for faster training and evaluation. 
   a. The file named ISIC segmentation will contain codes for the segmentation task and you could save those segmented images in a separate folder. 
   b. Now run the ISIC-Task-3 notebook to train and test models on the classification task. You could replace the segmented image folder in place of input image folder in the dataloaders. 

3. You could also look into the phase-I-report for a detailed info about our project.

## Conclusion
This project demonstrates the feasibility of developing an efficient and accurate skin lesion diagnosis system. The proposed methodologies ensure reduced runtime complexity and enhanced performance, making it suitable for practical clinical applications.

## Acknowledgments
- ISIC challenge for providing the datasets.
- Open-source libraries and frameworks that facilitated model development.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For further queries or collaborations, feel free to reach out at mugunddhan3@gmail.com.

