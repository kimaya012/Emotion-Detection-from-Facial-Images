# Emotion Detection from Facial Images
This project demonstrates how to train and evaluate a Convolutional Neural Network (CNN) to classify facial expressions using the CK+ dataset. It includes image preprocessing, model training, evaluation, and testing on unseen data.

## Dataset
The dataset used for this project can be downloaded from [Kaggle CK+ Dataset](https://www.kaggle.com/datasets/shawon10/ckplus)
We use the CK+ (Extended Cohn-Kanade) dataset, which includes labeled facial expression images for emotions such as:
- Anger
- Contempt
- Disgust
- Fear
- Happy
- Sadness 
- Surprise

## Features
- End-to-end training pipeline using PyTorch
- Random image testing from dataset
- Confusion matrix and classification report
- Model saving for deployment
- Visualization of misclassified examples

## Model Architecture

EmotionCNN

- (net): Sequential
- (0): Conv2d(1, 32, kernel_size=3, padding=1)
- (1): ReLU()
- (2): MaxPool2d(2)
- (3): Conv2d(32, 64, kernel_size=3, padding=1)
- (4): ReLU()
- (5): MaxPool2d(2)
- (6): Flatten()
- (7): Linear(9216, 128)
- (8): ReLU()
- (9): Linear(128, 7)

## Results
- Validation Accuracy: 98.98% (on the CK+ subset used)
- Detailed precision, recall, and F1-scores per class

## Requirements
- Python 3.10+
- PyTorch
- Torchvision
- Matplotlib
- Seaborn
- Scikit-learn

## How to Run
1. Clone the repository
2. Download the CK+ dataset and place it in the expected folder
3. Open the Emotion_Detection_CKplus.ipynb notebook
4. Run all cells step-by-step

## Model Saving
The model is saved as emotion_cnn.pth for reuse or deployment.

## Project Screenshots
- Accuracy
![Em1](https://github.com/user-attachments/assets/65548205-1e35-49ee-8326-4ef98c06ec10)
- Confusion Matrix
![Em2](https://github.com/user-attachments/assets/b26cc9a7-c1cf-45b5-a153-ffb7950995c0)
![Em3](https://github.com/user-attachments/assets/d5c98f9b-7c99-4b78-801b-f6d4ac772746)
- Test Results
![Em4](https://github.com/user-attachments/assets/81551842-8be5-484d-85b1-93ba3b7afde4)
- Misclassified example
![Em5](https://github.com/user-attachments/assets/94b3e1cc-c312-4dd6-b64d-2fd4137ef4fd)


