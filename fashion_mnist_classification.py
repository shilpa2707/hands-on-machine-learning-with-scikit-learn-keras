import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# import dataset
(train_images, train_label),(test_images,test_label)= fashion_mnist.load_data()

#normalize data
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32')/255

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Split training data into trainaing and validation dataset
train_images, val_images, train_label, val_label = train_test_split(train_images, train_label,test_size=0.2,random_state=42)

#CNN model

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#train model
history = model.fit(train_images, train_label, epochs=5, validation_data=(val_images, val_label))
print(f'Model accuracy: {history.history["accuracy"]}')

plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

#Model evaluation on test data
test_loss, test_acc = model.evaluate(test_images,  test_label, verbose=2)
print(f'Test accuracy: {test_acc}')

predictions = model.predict(test_images)

num_samples_to_display = 5
for i in range(num_samples_to_display):
    print(f"Actual Label: {test_label[i]}, Predicted Label: {np.argmax(predictions[i])}")
