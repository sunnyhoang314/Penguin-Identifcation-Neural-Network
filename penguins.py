# Sunny Hoang  
# 30170708
# Tutorial 5
# CPSC 383
# Spring 2025
# June 16th, 2025 

import tensorflow as tf
import numpy as np
import csv
import random

print("TensorFlow version:", tf.__version__)
print("\n\n Messages complete! Let's start... \n\n")

##############################################

FEATURE_VEC_SIZE = 13

# given a list with the features and label for a sample (i.e. a list of strings representing a row of the csv),
# this function converts it to a numeric feature vector (row vector) and an integer label
# returns the tuple (feature_vector, label)
def getDataFromSample(sample):
    species_str = sample[1].strip()
    island_str = sample[2].strip()
    culmen_length = float(sample[3])
    culmen_depth = float(sample[4])
    flipper_length = float(sample[5])
    body_mass = float(sample[6])
    sex_str = sample[7].strip()
    year_str = sample[8].strip()

    #species label encode
    species_map = {'Adelie': 0, 'Gentoo': 1, 'Chinstrap': 2}
    label = species_map[species_str]

    #island encoding
    if island_str == "Torgersen":
        island_encoding = [1, 0, 0]
    elif island_str == "Biscoe":
        island_encoding = [0, 1, 0]
    else: # Dream
        island_encoding = [0, 0, 1]

    #sex encoding
    if sex_str == "male":
        sex_encoding = [1, 0, 0]
    elif sex_str == "female":
        sex_encoding = [0, 1, 0]
    else: #NA
        sex_encoding = [0, 0, 1]

    #year encoding
    if year_str == "2007":
        year_encoding = [1, 0, 0]
    elif year_str == "2008":
        year_encoding = [0, 1, 0]
    else:
        year_encoding = [0, 0, 1]

    # combine all features
    feature_vector = np.array(
        island_encoding + 
        [culmen_length, culmen_depth, flipper_length, body_mass] + 
        sex_encoding +
        year_encoding
    )
    
    return (feature_vector, label)

##############################################

# reads number of samples, features, and their labels from the given file
# and converts lists of features to feature vectors
# returns (features_array, labels_array) as a tuple
# where features_array is a (n, FEATURE_VEC_SIZE) numpy array and labels_array is a (n,) numpy array

def readData(filename):
    with open(filename, newline='') as datafile:
        reader = csv.reader(datafile)
        next(reader, None) # skip the header row

        n = 0
        features = []
        labels = []

        for row in reader:
            # convert row to feature vector + numeric label
            feature_vector, label = getDataFromSample(row)
            
            # append to arrays of all data points and labels
            features.append(feature_vector)
            labels.append(label)
            
            # increment number of data points read
            n = n + 1

    print(f"Number of data points read: {n}")

    # convert to numpy arrays of the appropriate dimensions
    features_array = np.array(features)
    labels_array = np.array(labels)

    return (n, features_array, labels_array)

##############################################

# reads the data from the penguins.csv file,
# passes the rows to be encoded into feature vectors,
# and divides the data into training and testing sets
def prepData():
    n, features, labels = readData('penguins.csv')

    # Shuffle the data
    indices = np.arange(n)
    np.random.shuffle(indices)
    features = features[indices]
    labels = labels[indices]

    # 5:1 train-test split
    split_index = int(5/6 * n)

    trainingFeatures = features[:split_index]
    trainingLabels = labels[:split_index]
    testingFeatures = features[split_index:]
    testingLabels = labels[split_index:]

    trainingData = (trainingFeatures, trainingLabels)
    testingData = (testingFeatures, testingLabels)
    return (trainingData, testingData)

###################################################

# Creates a balanced model with regularization and normalization for high accuracy
def create_optimal_model():
    model = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(input_shape=(FEATURE_VEC_SIZE,)),  # Normalize input features
        tf.keras.layers.Dense(64, activation='relu'),  # First hidden layer
        tf.keras.layers.Dropout(0.2),  # Prevent overfitting
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation='relu'),  # Second hidden layer
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(16, activation='relu'),  # Third hidden layer
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(3, activation='softmax')  # Output layer (3 classes)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),  # Adaptive optimizer
        loss='sparse_categorical_crossentropy',  # For integer labels
        metrics=['accuracy']
    )

    return model

# Trains and evaluates the model
def train_and_evaluate_model(model, model_name, training_data, testing_data, epochs):
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    train_features, train_labels = training_data
    test_features, test_labels = testing_data

    # Normalize feature vectors
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    std = np.where(std == 0, 1, std)  # Prevent divide by zero
    train_features_norm = (train_features - mean) / std
    test_features_norm = (test_features - mean) / std

    # Use callbacks to improve training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.0001
        )
    ]

    # Train the model
    history = model.fit(
        train_features_norm, train_labels,
        epochs=epochs,
        validation_data=(test_features_norm, test_labels),
        callbacks=callbacks,
        verbose=1
    )

    # Final evaluation
    train_loss, train_accuracy = model.evaluate(train_features_norm, train_labels, verbose=0)
    test_loss, test_accuracy = model.evaluate(test_features_norm, test_labels, verbose=0)

    print(f"\n{model_name} Final Results:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Training Loss: {train_loss:.4f}")

    return train_accuracy, test_accuracy, test_loss

# Main function to run training and evaluation
def main():
    # Set seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print("Preparing data...")
    training_data, testing_data = prepData()
    
    print("Creating optimal model...")
    optimal_model = create_optimal_model()
    
    # Train and evaluate the optimal model
    train_acc, test_acc, test_loss = train_and_evaluate_model(
        optimal_model, "Optimal", training_data, testing_data, epochs=20
    )
    
    # Print final testing results
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print("Optimal Model:")
    print(f"  Testing Accuracy:  {test_acc:.4f}")
    print(f"  Testing Loss:      {test_loss:.4f}")

if __name__ == "__main__":
    main()
