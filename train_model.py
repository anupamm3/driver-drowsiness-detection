from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.model_arch import build_model
import os
import matplotlib.pyplot as plt

# Constants
TRAIN_DIR = 'data/train'
MODEL_PATH = 'models/drowsiness_cnn.h5'
ASSET_PATH = 'assets/training_graph.png'
EPOCHS = 25
BATCH_SIZE = 32

def train_model():
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('assets', exist_ok=True)
    
    print("Training Driver Drowsiness Detection CNN")
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    print("\nLoading training data...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(64, 64),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        subset='training',
        shuffle=True
    )

    print("\nLoading validation data...")
    validation_generator = validation_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(64, 64),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        subset='validation',
        shuffle=False
    )
    
    # Print class mapping
    print("Class indices:", train_generator.class_indices)
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")

    # Build the model
    print("\nBuilding model...")
    model = build_model()
    model.summary()

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks for better training
    callbacks = [
        # Save best model based on validation accuracy
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Stop training if validation loss doesn't improve for 5 epochs
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
    ]

    # Train the model
    print("Starting training...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate final model
    print("Evaluating model...")
    val_loss, val_accuracy = model.evaluate(validation_generator)
    print(f"Final Validation Loss: {val_loss:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")

    # Save final model
    model.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # Plot training & validation accuracy and loss
    print(f"\nGenerating training graphs...")
    plt.figure(figsize=(14, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ASSET_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training graphs saved to: {ASSET_PATH}")
    
if __name__ == '__main__':
    train_model()