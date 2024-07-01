import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split

# Data collection and processing
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
breast_cancer_dataset
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame.head()

#adding the target column to the data frame
data_frame['label'] = breast_cancer_dataset.target
data_frame.tail()
data_frame.shape
data_frame.info()
data_frame.isnull().sum()
data_frame.describe()

#cheaking the distribution of target variable
data_frame['label'].value_counts()
data_frame.groupby('label').mean()

#separating the features and target
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

#Spliting the data into training data and testing data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, x_train.shape, x_test.shape)

# Standardized the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_std = scaler.fit_transform(x_train)
x_test_std = scaler.transform(x_test)

#Building the Neural Network
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras

#setting up the layers of Neural Network
model = keras.Sequential([
        keras.layers.Flatten(input_shape=(30,)),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(2,activation='sigmoid')
    ])

# compiling the Neural Network
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

#training the neural network
history = model.fit(x_train, y_train, validation_split = 0.1, epochs = 10)

#Visualizing accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training data', 'validation data'], loc='lower right')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training data', 'validation data'], loc='upper right')

#Accuracy of the model on test data
loss, accuracy = model.evaluate(x_test_std, y_test)
accuracy

y_pred = model.predict(x_test_std)
y_pred.shape
y_pred

#model.predict() gives the prediction probability of each class for that data point
y_pred_labels = [np.argmax(i) for i in y_pred]

# Building the predictive system
input_data = ()
#change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)
#reshape the numpy array as we are predicting for one data point
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
#standardizing the input data
input_data_std = scaler.transform(input_data_reshaped)
prediction = model.predict(input_data_std)
print(prediction)

prediction_label = [np.argmax(prediction)]
print(prediction_label)
if(prediction_label[0]==0):
    print('The tumor is malignant')
else:
    print('The tumor is benign')





























