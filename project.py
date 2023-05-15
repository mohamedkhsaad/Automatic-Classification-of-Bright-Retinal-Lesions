import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score,classification_report
import time
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.model_selection import train_test_split

############################### preprocess the images ###########################
def preprocess_image(image_path, target_size):
    # Load and preprocess a single image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)  # Resize the image using bilinear interpolation
    image = image.astype('float32')/255.0 # Normalize the image pixel values to 0-1 scale

    # # Apply contrast enhancement using histogram equalization
    # image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # image_eq = cv2.equalizeHist(image_gray)
    # image_eq_rgb = cv2.cvtColor(image_eq, cv2.COLOR_GRAY2RGB)

    return image
############################### Load the images ###############################
def load_images_from_directory(directory, target_size):
    image_list = []
    label_list = []

    # Specify the class labels
    class_labels = ["normal", "exudates", "drusen"]

    # Iterate over the subdirectories (classes) in the directory
    for class_name in class_labels:
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            # Iterate over the images in the class directory
            for filename in os.listdir(class_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust the file extensions as needed
                    image_path = os.path.join(class_dir, filename)
                    image = preprocess_image(image_path, target_size)
                    image_list.append(image)
                    label_list.append(class_name)

    # Convert the image list and label list to NumPy arrays
    image_array = np.array(image_list)
    label_array = np.array(label_list)

    return image_array, label_array, class_labels

# Set the directory path where your images are located
data_directory = "E:\Spring 23\SBEN424 - Advanced Image processing\Project"
# Set the target size for resizing the images
target_size = (256, 256)  # Adjust the size as needed

# Load and preprocess the images from the directory
image_array, label_array,class_labels = load_images_from_directory(data_directory, target_size)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_array, label_array, test_size=0.4, random_state=42)

# Now you have the training and testing sets (X_train, X_test) and their corresponding labels (y_train, y_test)

#Starting time
start=time.time()

################################# Training the CNN model #################################

# Convert the string labels to numeric format
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)


# Define the CNN model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
                    #    'specificity','precision','f1','AUC',
                    #    'Recall','FalsePositiveRate','TruePositiveRate','FalsePositiveRate','TrueNegativeRate',
                    #    'FalseNegativeRate','ConfusionMatrix','robustness'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

#Ending time
end=time.time()

############################### Evaluate the model on the test data ###############################
# Make predictions on the test set
probabilities = model.predict(X_test)
y_pred = probabilities.argmax(axis=1)  # Convert probabilities to class labels

# Convert predictions and true labels back to their original form
y_pred = label_encoder.inverse_transform(y_pred)
y_test = label_encoder.inverse_transform(y_test)
# Calculate AUC
auc = roc_auc_score(y_test, probabilities, multi_class='ovr')

#Transforming the labels to binary #Failed
# y_pred[y_pred=='normal']=0
# y_pred[y_pred=='exudates']=1
# y_pred[y_pred=='drusen']=1
# y_test[y_test=='normal']=0
# y_test[y_test=='exudates']=1
# y_test[y_test=='drusen']=1

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate sensitivity (recall)
sensitivity = recall_score(y_test, y_pred, average='macro')

# Calculate specificity
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    return specificity

specificity = specificity_score(y_test, y_pred)

# Calculate precision
precision = precision_score(y_test, y_pred, average='macro')

# Calculate F1-score
f1 = f1_score(y_test, y_pred, average='macro')

# Print the metrics
print("Accuracy:", accuracy)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("Precision:", precision)
print("F1-score:", f1)
print("AUC:", auc)
print("Time:",(end-start))
print(classification_report(y_test, y_pred, target_names=class_labels))


# # Make predictions on new data
# new_data = [...]  # Replace [...] with your new data
# preprocessed_data = [...]  # Preprocess the new data
# predictions = model.predict(preprocessed_data)

# # Print the predicted class labels
# predicted_labels = np.argmax(predictions, axis=1)
# class_labels = ["normal", "exudates", "drusen"]
# for label in predicted_labels:
#     print(class_labels[label])





