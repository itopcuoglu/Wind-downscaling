# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:44:14 2025

@author: uegerer
"""

# !pip install tensorflow


import tensorflow as tf
from tensorflow.keras import layers, models



#%% Define the 3D CNN Model


"""
CNN Architecture Design

A simple 3D CNN architecture could look like this:

    Input Layer: Your input will be a 4D tensor with shape (batch_size, time_steps, lat, lon, features).

    Conv3D Layers:
        Apply 3D convolutions to learn spatial and temporal patterns.
        A good starting point would be:
            Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu'): 3D filter over time, latitude, and longitude.
            Use batch normalization after convolutions to stabilize learning.
            Add dropout after each layer to prevent overfitting (optional).

    MaxPooling3D: Apply 3D pooling to reduce the spatial and temporal dimensions.
        Example: MaxPooling3D(pool_size=(2, 2, 2)) reduces the dimensions in all three axes.

    Flatten Layer: After the convolutions, flatten the output to a 1D vector, ready for fully connected layers.

    Dense Layers: After flattening, apply fully connected layers:
        Example: Dense(64, activation='relu')
        Optionally, you can add more layers or dropout to regularize the network.

    Output Layer: If you're predicting a time series, the output shape might be (batch_size, num_predictions) or a grid of predictions for each lat-lon location.



Things to Consider

    Data Preprocessing:
        Normalize your input data (e.g., wind speed values, temperature) before feeding it into the network.
        Consider downsampling your input data if it’s too large (in terms of lat-lon grid size or time steps).

    Model Tuning:
        Experiment with different numbers of layers and filters (32, 64, 128, etc.) depending on the complexity of your problem.
        Adjust kernel sizes, pooling sizes, and dropout rates based on model performance.

    Evaluation:
        Monitor validation loss and MAE during training to avoid overfitting and make sure your model generalizes well to unseen data.


"""
    

def create_3d_cnn(input_shape):
    model = models.Sequential()

    # 3D Convolutional Layer
    model.add(layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    # Second 3D Convolutional Layer
    model.add(layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    # Third 3D Convolutional Layer
    model.add(layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    # Flatten the 3D output into 1D for Dense layers
    model.add(layers.Flatten())

    # Fully Connected Layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout for regularization
    model.add(layers.Dense(64, activation='relu'))

    # Output Layer (e.g., predicting wind speed at new lat-lon points)
    model.add(layers.Dense(1))  # Output a single continuous value per sample (wind speed)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    return model




#%% Train the model


# Example data shapes (you need to load your actual data)
input_shape = (30, 100, 100, 3)  # 30 time steps, 100x100 grid, 3 features (e.g., wind speeds)
X_train = # Load your training data (shape: (num_samples, time_steps, lat, lon, features))
y_train = # Load your target data (shape: (num_samples, 1)) - wind speed predictions

# Create the model
model = create_3d_cnn(input_shape)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)




#%% Evaluation and prediction

# Example test data
X_test = # Load your test data
y_test = # Load your test labels

# Evaluate the model on test data
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Make predictions
y_pred = model.predict(X_test)




#%% Save and load the model


# Save the trained model
model.save("wind_prediction_model.h5")

# To load the model later
loaded_model = tf.keras.models.load_model("wind_prediction_model.h5")


























