"""
Shared utility functions for the German Credit project.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    import random
    
    np.random.seed(seed)
    random.seed(seed)
    
    # Try to set TensorFlow seed if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass  # TensorFlow not installed, skip
    
    return seed


def print_class_distribution(y, label=""):
    """Print class distribution for classification tasks."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    print(f"\n{label} Class Distribution:")
    for cls, count in zip(unique, counts):
        percentage = (count / total) * 100
        credit_type = "Good Credit" if cls == 1 else "Bad Credit"
        print(f"  Class {cls} ({credit_type}): {count} samples ({percentage:.1f}%)")
    
    return dict(zip(unique, counts))


def calculate_class_weights(y):
    """Calculate balanced class weights for imbalanced datasets."""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """Generate and optionally save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Good (1)', 'Bad (2)'],
                yticklabels=['Good (1)', 'Bad (2)'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    return cm


def print_model_summary(model_name, metrics_dict):
    """Print a formatted summary of model performance."""
    print(f"\n{'='*50}")
    print(f"{model_name} Performance Summary")
    print(f"{'='*50}")
    
    for metric_name, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"{metric_name:20s}: {value:.4f}")
        else:
            print(f"{metric_name:20s}: {value}")
    
    print(f"{'='*50}")


def check_for_data_leakage(X_train, X_test, threshold=0.95):
    """Check for potential data leakage between train and test sets."""
    from sklearn.metrics.pairwise import cosine_similarity
    
     Sample if datasets are large
    n_samples = min(100, len(X_train), len(X_test))
    
    if hasattr(X_train, 'toarray'):  # Handle sparse matrices
        X_train_sample = X_train[:n_samples].toarray()
        X_test_sample = X_test[:n_samples].toarray()
    else:
        X_train_sample = X_train[:n_samples]
        X_test_sample = X_test[:n_samples]
    
    similarities = cosine_similarity(X_test_sample, X_train_sample)
    max_similarities = similarities.max(axis=1)
    
    # Check for suspiciously high similarities
    suspicious = (max_similarities > threshold).sum()
    
    if suspicious > 0:
        print(f"⚠️ Warning: {suspicious} test samples have >={threshold:.0%} similarity with training data")
        print("This might indicate data leakage!")
    else:
        print(f"✓ No obvious data leakage detected (threshold: {threshold:.0%})")
    
    return max_similarities


def save_metrics_to_csv(metrics_dict, filepath):
    """Save metrics dictionary to CSV file."""
    df = pd.DataFrame([metrics_dict])
    df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")
    return df