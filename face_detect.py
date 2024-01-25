import os
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Define the path to the folder containing the images
image_folder = "D:/HUST/dev/py/face/hoang_face_2"
csv_file = 'D:/HUST/dev/py/face/face_data.csv'

# Load the CSV file using pandas
df = pd.read_csv(csv_file)

# Retrieve image file names and labels
image_files = df['Image file'].tolist()
labels = df['Label'].tolist()

# Define the batch size and number of epochs for training
batch_size = 16
epochs = 10

# Load and preprocess the images
total_images = len(image_files)
input_shape = (32, 32, 3)  # Assuming images are RGB and have size 32x32

# Create empty arrays to store the images and labels
images = np.empty((total_images, *input_shape))

for i, file in enumerate(image_files):
    img_path = os.path.join(image_folder, file)
    img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    img_array = image.img_to_array(img)
    images[i] = img_array

# Normalize the image data
images = images / 255.0

# Convert labels to numerical format
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(labels)
onehot_encoder = OneHotEncoder(sparse=False)
onehot_labels = onehot_encoder.fit_transform(integer_labels.reshape(-1, 1))
num_classes = len(label_encoder.classes_)

# Split the data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(
    images, onehot_labels, test_size=0.2, random_state=42
)

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_images, test_labels))

# Save the trained model
model.save("face.keras")
print("Model saved.")

