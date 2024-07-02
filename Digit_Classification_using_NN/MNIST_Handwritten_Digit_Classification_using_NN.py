# Importing the dependencies
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist
from  tensorflow.math import confusion_matrix

#Loading the MNIST data from keras dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
type(X_train)

#shape of the numpy arrays
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

# Training data = 60,000 images      Test data = 10,000 images
# Image dimension = 28 * 28          Grayscale image = 1 channel
print(X_train[10].shape)

#displaying the image
plt.imshow(X_train[25])
plt.show()
print(Y_train[25])                  #print the corresponding label

#Image Labels
print(Y_train.shape, Y_test.shape)
print(np.unique(Y_train))           #unique value in Y_train
print(np.unique(Y_test))            #unique value in Y_test

# We can use these labels as such or we can also apply one hot encoding
#All the images have the same dimensions in this dataset. If not we have to resize all the images to a common dimension
X_train = X_train/255
X_test = X_test/255

print(X_train[10])

#Building the Neural Network
#setting up the layers of the NN
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
    ])

#compiling the NN
model.compile(optimizer='adam',
              loss= 'sparse_categorical_crossentropy',
              metrics = ['accuracy']
              )

# Training the NN
model.fit(X_train, Y_train, epochs=10)     #Training data accuracy = 98.9%

#Accuracy on test data
loss, accuracy = model.evaluate(X_test, Y_test)
print(accuracy)                             #Training data accuracy = 97.3%

#first data point in x_test
plt.imshow(X_test[0])
plt.show()
print(Y_test[0])

Y_pred = model.predict(X_test)
Y_pred.shape
Y_pred[0]

#converting the prediction probabilities to class label
label_for_first_test_image = np.argmax(Y_pred[0])
print(label_for_first_test_image)

#converting the prediction probabilities to class label for all test data points
Y_pred_labels = [np.argmax(i) for i in Y_pred]
print(Y_pred_labels)                          #Predicted label

#Confusion Matrix
conf_mat = confusion_matrix(Y_test, Y_pred_labels)
print(conf_mat)

plt.figure(figsize=(15,7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')

#Building a predictive system
input_image_path = 'C:/Users/Dell/Desktop/Deep Learning Projects/Digit_Classification_using_NN/sample_img.png'
input_img = cv2.imread(input_image_path)
type(input_img)
print(input_img)
cv2.imshow('Image Window',input_img)
input_img.shape

grayscale = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
grayscale.shape

input_img_resize = cv2.resize(grayscale,(28,28))
input_img_resize.shape
cv2.imshow('image window',input_img_resize)

input_img_resize = input_img_resize/255
image_reshape = np.reshape(input_img_resize, [1, 28, 28])
input_prediction = model.predict(image_reshape)
print(input_prediction)

input_pred_label = np.argmax(input_prediction)
print(input_pred_label)


#Predictive System

input_image_path = input("Path of the image to be predicted: ")
input_img = cv2.imread(input_image_path)
cv2.imshow('Image Window',input_img)
grayscale = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
input_img_resize = cv2.resize(grayscale,(28,28))
input_img_resize = input_img_resize/255
image_reshape = np.reshape(input_img_resize, [1, 28, 28])
input_prediction = model.predict(image_reshape)
input_pred_label = np.argmax(input_prediction)
print("The handwritten digit is recognized as ",input_pred_label)






