!pip install kaggle

# configuring the path of kaggle.json file
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Importing the DOG vs CAT dataset from Kaggle
!kaggle competitions download -c dogs-vs-cats

!ls

# extracting the compressed dataset
from zipfile import ZipFile
dataset = "dogs-vs-cats.zip"

with ZipFile(dataset, 'r') as zip:
  zip.extractall()
  print("The dataset is extracted")

# extracting the compressed dataset
from zipfile import ZipFile
dataset = "train.zip"

with ZipFile(dataset, 'r') as zip:
  zip.extractall()
  print("The dataset is extracted")

import os
# counting the number of files in train folder
path, dirs, files = next(os.walk('/content/train'))
file_count = len(files)
print("Number of images", file_count)

#Printing the name of images
file_names = os.listdir('/content/train')
print(file_names)

#Importing the dependencies
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from google.colab.patches import cv2_imshow

# display cat image
img = mpimg.imread('/content/train/cat.1938.jpg')
imgplt = plt.imshow(img)
plt.show()

# display dog image
img = mpimg.imread('/content/train/dog.8965.jpg')
imgplt = plt.imshow(img)
plt.show()

file_names = os.listdir('/content/train')
dog_count = 0
cat_count = 0

for img_file in file_names:
  name = img_file[0:3]

  if name =='dog':
    dog_count += 1
  else:
    cat_count += 1

print("Number of dog images : ", dog_count)
print("Number of cat images : ", cat_count)

# Resizing all the images
os.mkdir('/content/image_resized')

original_folder = '/content/train'
resized_folder = '/content/image_resized'

for i in range(2000):

  filename = os.listdir(original_folder)[i]
  img_path = os.path.join(original_folder, filename)

  img = Image.open(img_path)
  img = img.resize((224,224))
  img = img.convert('RGB')

  newImgPath = os.path.join(resized_folder, filename)
  img.save(newImgPath)

# display resized cat image
img = mpimg.imread('/content/image_resized/cat.1938.jpg')
imgplt = plt.imshow(img)
plt.show()

# display resized dog image
img = mpimg.imread('/content/image_resized/dog.8965.jpg')
imgplt = plt.imshow(img)
plt.show()

# Creating labels for resized images of dogs and cats
# Cat --> 0
# Dog --> 1

filenames = os.listdir('/content/image_resized/')

labels = []

for i in range(2000):
  file_name = filenames[i]
  label = file_name[0:3]

  if label == 'dog':
    labels.append(1)
  else:
    labels.append(0)

print(filenames[0:5])
print(labels[0:5])

# counting the images of dogs and cats out of 2000 images
values, count = np.unique(labels, return_counts=True)
print(values)
print(count)

# Converting all the resized images to the numpy array
import cv2
import glob

image_directory = "/content/image_resized/"
image_extension = ['png', 'jpg']

files = []
[files.extend(glob.glob(image_directory + '*.' + e)) for e in image_extension]

dog_cat_images = np.asarray([cv2.imread(file) for file in files])

print(dog_cat_images)

type(dog_cat_images)

print(dog_cat_images.shape)

X = dog_cat_images
Y = np.asarray(labels)

# Train test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, x_train.shape, x_test.shape)

# Scaling the data
x_train_scaled = x_train/255
x_test_scaled = x_test/255

print(x_train_scaled)

# Building the Neural Network
import tensorflow as tf
import tensorflow_hub as hub

mobilenet_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4'

pretrained_model = hub.KerasLayer(mobilenet_model, input_shape=(224,224,3),trainable=False)

num_of_classes = 2

model = tf.keras.Sequential([

    pretrained_model,
    tf.keras.layers.Dense(num_of_classes)

])

model.summary()

model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['acc']
)

model.fit(x_train_scaled, y_train, epochs=5)

score, acc = model .evaluate(x_test_scaled, y_test)
print("Test loss = ",score)
print("Test Accuracy = ", acc)

# Predictive System

input_image_path = input('Path of the image to be predicted: ')
input_image = cv2.imread(input_image_path)
cv2_imshow(input_image)
input_image_resize = cv2.resize(input_image, (224,224))
input_image_scaled = input_image_resize/255
image_reshaped = np.reshape(input_image_scaled, [1,224,224,3])
input_prediction = model.predict(image_reshaped)
input_pred_label = np.argmax(input_prediction)

if input_pred_label == 0:
  print('The image represents a cat')
else:
  print('The image represents a dog')