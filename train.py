# Import modules
import os
import argparse
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow

# Enable autologging for mlflow
mlflow.tensorflow.autolog()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
args = parser.parse_args()

# Set hyperparameters
learning_rate = args.learning_rate
batch_size = args.batch_size
epochs = args.epochs

# Load the MNIST dataset from zip file
with np.load("mnist/mnist.zip") as data:
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

# Normalize and reshape the data
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Create the CNN model
def create_model():
    # Create a sequential model
    model = tf.keras.Sequential()

    # Add a convolutional layer with 32 filters, 3x3 kernel size, ReLU activation, and input shape of 28x28x1
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))

    # Add a max pooling layer with 2x2 pool size
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # Add another convolutional layer with 64 filters, 3x3 kernel size, and ReLU activation
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))

    # Add another max pooling layer with 2x2 pool size
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # Add a flatten layer
    model.add(tf.keras.layers.Flatten())

    # Add a dense layer with 10 units and softmax activation
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    # Return the model
    return model

# Create the CNN model
model = create_model()

# Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

# Train the model on the training data and validate on the test data
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

# Evaluate the model on the test data and print the results
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_acc}")

# Save the model as a TensorFlow SavedModel
model.save("outputs/model")
