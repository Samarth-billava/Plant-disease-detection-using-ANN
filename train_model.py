import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset path
DATASET_PATH = r"C:\Users\samar\Desktop\Contriver\Plant_disease_prediction\dataset"

# Image preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    DATASET_PATH, target_size=(128, 128), batch_size=32, class_mode='categorical', subset='training')

val_generator = datagen.flow_from_directory(
    DATASET_PATH, target_size=(128, 128), batch_size=32, class_mode='categorical', subset='validation')

# Define ANN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Save the model
model.save("plant_disease_model.h5")

print("Model training complete! Saved as 'plant_disease_model.h5'")
