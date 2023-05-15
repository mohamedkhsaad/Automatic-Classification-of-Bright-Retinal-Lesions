# Retinal Image Analysis

This project focuses on the analysis of retinal images for eye disease detection using deep learning techniques.

## Dataset

The dataset used for this project consists of retinal images categorized into three classes: normal, drusen, and exudates. The dataset is split into training and testing sets, with an 80:20 ratio.

## Data Preprocessing

- Resizing: All retinal images are resized to a fixed size for consistency.
- Normalization: Normalization techniques are applied to enhance image quality.
- Augmentation: Data augmentation techniques such as rotation and flipping are used to increase the diversity of the training data.

## Model Architecture

The model used for this project is a Convolutional Neural Network (CNN) with the following architecture:

- Multiple convolutional and pooling layers.
- Dropout layers for regularization and reducing overfitting.
- Final dense layers with softmax activation for classification.

## Training and Evaluation

The model is trained for 10 epochs with a batch size of 32. The training process achieves an accuracy of 79.82% and a validation accuracy of 84.93%. The model is then evaluated on the test set, resulting in an accuracy of 84.93%.

## Evaluation Metrics

The following evaluation metrics are calculated:

- Accuracy: 84.93%
- Sensitivity: 85.33%
- Specificity: 100%
- Precision: 90.60%
- F1-score: 85.13%
- AUC: 97.72%

## ROC Curve

The Receiver Operating Characteristic (ROC) curve is plotted to visualize the model's performance. The curve shows the trade-off between the true positive rate (TPR) and the false positive rate (FPR).

## Classification Report

A classification report is generated to provide a detailed evaluation of the model's performance for each class (normal, drusen, exudates). The report includes precision, recall, f1-score, and support for each class.

## Robustness and Computational Time

To ensure model robustness, k-fold cross-validation is applied for performance evaluation. Hyperparameter tuning is conducted to optimize model performance.

The computational time required for training and inference is approximately 51.11 seconds.

## Summary and Conclusion

The developed CNN model shows promising accuracy in classifying retinal images into normal, drusen, and exudates categories. The model demonstrates potential for assisting in automated eye disease detection. However, the limitations of a limited dataset size and room for further exploration of advanced architectures and techniques should be considered for future improvements.


