import tensorflow as tf
import os
from data_loader import get_data_generators
from swin_model import build_swin_model

# Load data
train_generator, val_generator, _ = get_data_generators()

# Build model
model = build_swin_model()

# Training parameters
EPOCHS = 10
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "../models/best_model.keras")

# Checkpoint callback to save the best model
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    CHECKPOINT_PATH, save_best_only=True, monitor='val_accuracy', mode='max'
)

# Train model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb]
)

print("Training Complete. Best model saved at:", CHECKPOINT_PATH)
