# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
The goal of this project is to develop a convolutional neural network (CNN) for classifying handwritten digits from the MNIST dataset. The dataset contains 70,000 grayscale images of digits (0-9) with each image sized at 28x28 pixels. The objective is to accurately predict the digit in each image, thereby demonstrating the effectiveness of CNNs for image classification tasks.

![image](https://github.com/user-attachments/assets/6d73f804-6b79-4e03-95c9-9809147df701)


## Neural Network Model

![image](https://github.com/user-attachments/assets/14e480ae-f057-432c-9495-8be9f4651e88)


### DESIGN STEPS

#### Step 1: 
Load the MNIST dataset of handwritten digits.

#### Step 2: 
Normalize images and convert labels to one-hot encoding.

#### Step 3: 
Build and compile a convolutional neural network (CNN).

#### Step 4: 
Train the CNN on the training dataset with validation.

#### Step 5: 
Evaluate model performance using accuracy and classification metrics.

#### Step 6: 
Make predictions on new images and visualize the results.

## PROGRAM

### Name: Bhargava S
### Register Number: 212221040029
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Explore the shape of the dataset
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Visualize a single image
single_image = X_train[0]
plt.imshow(single_image, cmap='gray')
plt.title("Single Image")
plt.show()
# Scale the images
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

# Convert labels to one-hot encoding
y_train_onehot = utils.to_categorical(y_train, 10)
y_test_onehot = utils.to_categorical(y_test, 10)

# Reshape the data to fit the model input
X_train_scaled = X_train_scaled.reshape(-1, 28, 28, 1)
X_test_scaled = X_test_scaled.reshape(-1, 28, 28, 1)
# Define the CNN model
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Display the model summary
model.summary()
# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train_onehot, epochs=5,
                    batch_size=64, validation_data=(X_test_scaled, y_test_onehot))
# Convert the history to a DataFrame
metrics = pd.DataFrame(history.history)

# Plot accuracy and loss
metrics[['accuracy', 'val_accuracy']].plot()
plt.title("Model Accuracy")
plt.show()

metrics[['loss', 'val_loss']].plot()
plt.title("Model Loss")
plt.show()

# Make predictions on the test set
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

# Confusion matrix and classification report
print(confusion_matrix(y_test, x_test_predictions))
print(classification_report(y_test, x_test_predictions))
# Load and preprocess a new image
img = image.load_img('/content/five.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor, (28, 28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy() / 255.0

# Predict the class of the new image
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1, 28, 28, 1)),
    axis=1
)

print('Prediction for img: ', x_single_prediction)

# Visualize the new image
plt.imshow(img_28_gray_scaled.reshape(28, 28), cmap='gray')
plt.title("Processed Image")
plt.show()

# Invert the image colors and predict again
img_28_gray_inverted = 255.0 - img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy() / 255.0

x_single_prediction_inverted = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1, 28, 28, 1)),
    axis=1
)

print("Inverted prediction: ", x_single_prediction_inverted)

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/3f6f3faa-098c-4c6a-9b6d-61708f5bb0e3)


### Classification Report

![image](https://github.com/user-attachments/assets/1f56d547-4c2d-4772-a7ee-fbc29ccbeead)



### Confusion Matrix

![image](https://github.com/user-attachments/assets/754cb6f6-5106-42ca-be8e-b061a9020c98)


### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/2cf0f177-f711-4d16-88fe-7958a78d17fe)

## RESULT
The model achieved a high accuracy of 99% on the MNIST test dataset. The classification report shows that the model performed exceptionally well across all digit classes, with precision, recall, and F1-scores all being close to 1. This indicates that the CNN model is effective for digit classification tasks in this context.
