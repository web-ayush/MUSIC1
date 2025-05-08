import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam

# Path to the genres folder
DATA_PATH = "genres"

# List of all genres (subfolder names)
genres = os.listdir(DATA_PATH)

# Function to extract features from audio
def extract_features(file_path):
    """
    Extract MFCC features from an audio file.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        np.ndarray: Mean MFCC features.
    """
    y, sr = librosa.load(file_path, mono=True, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Prepare data and labels
X = []
y = []

for idx, genre in enumerate(genres):
    genre_path = os.path.join(DATA_PATH, genre)
    if not os.path.isdir(genre_path):
        continue
    for filename in os.listdir(genre_path):
        file_path = os.path.join(genre_path, filename)
        try:
            features = extract_features(file_path)
            X.append(features)
            y.append(idx)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Convert to numpy arrays
X = np.array(X)
y = to_categorical(np.array(y), num_classes=len(genres))

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Input(shape=(13,)),  # Input layer with 13 features (MFCCs)
    Dense(100, activation='relu'),  # First hidden layer
    Dense(50, activation='relu'),   # Second hidden layer
    Dense(len(genres), activation='softmax')  # Output layer with softmax activation
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Check if model has weights and evaluate the model
if model.count_params() > 0:
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.2f}")

    # Save the model in Keras format
    try:
        model.save("music_genre_model.keras")
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")
else:
    print("Model has no trainable parameters, skipping save.")

# Load the model back to check if it's saved and working correctly
try:
    loaded_model = load_model('music_genre_model.keras')
    print("Model loaded successfully!")
    
    # Print the model architecture
    loaded_model.summary()

    # Evaluate the loaded model
    test_loss, test_acc = loaded_model.evaluate(X_test, y_test)
    print(f"Test Accuracy of Loaded Model: {test_acc:.2f}")
except Exception as e:
    print(f"Error loading model: {e}")
