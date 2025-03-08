import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from data_loader import get_data_generators

# Load test data
_, _, test_generator = get_data_generators()

# Load the best trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/best_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

# Get true labels
y_true = test_generator.classes

# Predict
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Compute performance metrics
auc = roc_auc_score(y_true, y_pred_prob)
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn + fp)
sensitivity = recall

# Save and print metrics
metrics_df = pd.DataFrame({
    "Metric": ["AUC", "Accuracy", "Precision", "Recall", "F1 Score", "Specificity"],
    "Value": [auc, acc, prec, recall, f1, specificity]
})

metrics_path = os.path.join(os.path.dirname(__file__), "../output/metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

print("\nPerformance Metrics:")
print(metrics_df)
print("\nMetrics saved at:", metrics_path)
