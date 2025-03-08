import tensorflow as tf
from tensorflow import keras
from keras_cv.models import SwinTransformerV2
from tensorflow.keras import layers, models

def build_swin_model():
    """Builds a Swin Transformer-based classification model for brain tumor detection."""

    # Load pre-trained Swin Transformer V2 model
    base_model = SwinTransformerV2(
        input_shape=(224, 224, 3),
        include_top=False,
        pretrained="imagenet",
    )

    # Freeze base model for transfer learning
    base_model.trainable = False

    # Classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
