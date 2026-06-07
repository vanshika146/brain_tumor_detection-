# Brain Tumor Detection using Machine Learning

## Overview

This project is a Brain Tumor Detection System that classifies MRI brain scans as either **Tumor** or **No Tumor** using Machine Learning techniques. The system preprocesses MRI images, trains multiple classification models, and provides predictions through a simple graphical user interface (GUI).

## Features

* MRI image preprocessing and normalization
* Binary classification of brain MRI scans
* Comparison of multiple machine learning algorithms
* Random Forest-based prediction model
* User-friendly GUI for image upload and prediction
* Model persistence using Joblib

## Dataset Information

The dataset consists of brain MRI images categorized into:

* Tumor (Label = 1)
* No Tumor (Label = 0)

### Preprocessing Steps

* Images converted to grayscale
* Images resized to 200 × 200 pixels
* Pixel values normalized to the range [0, 1]
* Images flattened into feature vectors for machine learning models

## Technologies Used

* Python
* NumPy
* Pandas
* OpenCV
* Matplotlib
* Seaborn
* Scikit-learn
* Tkinter
* Joblib

## Machine Learning Models Evaluated

The following models were trained and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM)

Based on performance evaluation, the **Random Forest Classifier** was selected as the final model.

## Results

| Metric            | Value   |
| ----------------- | ------- |
| Training Accuracy | 100.00% |
| Testing Accuracy  | 95.30%  |

The model achieved strong performance in distinguishing tumor and non-tumor MRI images.

## Project Workflow

1. Load MRI image dataset
2. Preprocess images
3. Convert images into feature vectors
4. Split data into training and testing sets
5. Train machine learning models
6. Evaluate model performance
7. Save trained model
8. Predict MRI images using GUI

## Project Structure

```text
brain_tumor_detection/
│
├── Training/
├── Testing/
├── brain_tumor_model.joblib
├── brain.ipynb
├── app.py
├── README.md
└── requirements.txt
```

## How to Run

### Clone Repository

```bash
git clone https://github.com/vanshika146/brain_tumor_detection-.git
cd brain_tumor_detection-
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

### Using the GUI

1. Launch the application.
2. Click **Upload MRI**.
3. Select an MRI image.
4. View the prediction result.


## Future Improvements

* Use Convolutional Neural Networks (CNNs) for feature extraction
* Expand dataset size for better generalization
* Deploy the model as a web application
* Support multi-class tumor classification

## Author

**Vanshika**
B.Tech Information Technology, IGDTUW

GitHub: https://github.com/vanshika146
