import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as nps

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Split the training data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Visualize an example image from the MNIST dataset
mnist_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(1)
for digit_images, labels in mnist_train.take(1):
    # Loop through the batch to access individual images
    for digit_image, label in zip(digit_images, labels):
        some_digit = digit_image.numpy().reshape((28, 28))
        plt.imshow(some_digit, cmap='binary', interpolation="nearest")
        plt.axis("off")
        plt.show()
        print("Class ID: %s and Class name: %s" % (label.numpy(), label.numpy()))

# Define the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with the training and validation data
history = model.fit(train_images, train_labels, epochs=5, validation_data=(val_images, val_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

predictions = model.predict(test_images)

num_samples_to_display = 5
for i in range(num_samples_to_display):
    print(f"Actual Label: {test_labels[i]}, Predicted Label: {np.argmax(predictions[i])}")