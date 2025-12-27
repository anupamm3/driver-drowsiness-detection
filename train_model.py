from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from src.model_arch import build_model
import os
import matplotlib.pyplot as plt

# Constants
TRAIN_DIR = 'data/train'
MODEL_PATH = 'models/drowsiness_cnn.h5'
ASSET_PATH = 'assets/training_graph.png'

def train_model():
    # Create ImageDataGenerator for training and validation
    datagen = ImageDataGenerator(validation_split=0.2)

    # Load training data
    train_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(64, 64),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=32,
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(64, 64),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=32,
        subset='validation'
    )

    # Build the model
    model = build_model()

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=10,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_steps=validation_generator.samples // validation_generator.batch_size
    )

    # Save the model
    model.save(MODEL_PATH)

    # Plot training & validation accuracy and loss
    plt.figure(figsize=(12, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Save the plots
    plt.savefig(ASSET_PATH)
    plt.close()

if __name__ == '__main__':
    train_model()