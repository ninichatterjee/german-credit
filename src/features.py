"""
Feature preprocessing for German Credit dataset.
Supports: Logistic Regression, Linear Regression, Decision Tree Classifier, Decision Tree Regressor
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

RANDOM_SEED = 42

# Define categorical and numerical feature indices based on German Credit dataset
# Categorical features: Attributes 1, 3, 4, 6, 7, 9, 10, 12, 14, 15, 17, 19, 20
CATEGORICAL_INDICES = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
# Numerical features: Attributes 2, 5, 8, 11, 13, 16, 18
NUMERICAL_INDICES = [1, 4, 7, 10, 12, 15, 17]


def handle_class_imbalance(X, y):
    """
    Handles class imbalance via undersampling majority class.
    Args:
        X (np.ndarray): Features
        y (np.ndarray): Target (1=Good, 2=Bad)
    
    Returns:
        X_clean (np.ndarray): Balanced features
        y_clean (np.ndarray): Balanced target
    """
    # Separate majority and minority classes
    # Class 1 (Good credit) = 700 samples, Class 2 (Bad credit) = 300 samples
    idx_class_1 = np.where(y == 1)[0]  # Majority class
    idx_class_2 = np.where(y == 2)[0]  # Minority class
    
    # Undersample majority class to match minority class
    np.random.seed(RANDOM_SEED)
    idx_class_1_downsampled = np.random.choice(
        idx_class_1, 
        size=len(idx_class_2), 
        replace=False
    )
    
    # Combine indices
    balanced_indices = np.concatenate([idx_class_1_downsampled, idx_class_2])
    
    # Shuffle indices
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(balanced_indices)
    
    # Return balanced data
    X_clean = X[balanced_indices]
    y_clean = y[balanced_indices]
    
    return X_clean, y_clean


def encode_categorical_features(X_train, X_val, X_test):
    """
    Encode categorical features using LabelEncoder.
    Fits on training data and transforms all sets.
    
    Args:
        X_train, X_val, X_test (np.ndarray): Feature arrays
    
    Returns:
        X_train_enc, X_val_enc, X_test_enc (np.ndarray): Encoded features
        encoders (dict): Dictionary of fitted encoders for each categorical column
    """
    X_train_enc = X_train.copy()
    X_val_enc = X_val.copy()
    X_test_enc = X_test.copy()
    
    encoders = {}
    
    # First, convert numerical columns to float (they're currently strings)
    for idx in NUMERICAL_INDICES:
        X_train_enc[:, idx] = X_train_enc[:, idx].astype(float)
        X_val_enc[:, idx] = X_val_enc[:, idx].astype(float)
        X_test_enc[:, idx] = X_test_enc[:, idx].astype(float)
    
    # Then, encode categorical columns
    for idx in CATEGORICAL_INDICES:
        encoder = LabelEncoder()
        
        # Fit on training data
        X_train_enc[:, idx] = encoder.fit_transform(X_train[:, idx])
        
        # Transform validation and test, handling unseen labels
        for X_set, X_enc_set in [(X_val, X_val_enc), (X_test, X_test_enc)]:
            # Handle unseen labels by mapping them to a new category
            labels = X_set[:, idx]
            encoded = np.zeros(len(labels), dtype=int)
            
            for i, label in enumerate(labels):
                if label in encoder.classes_:
                    encoded[i] = encoder.transform([label])[0]
                else:
                    # Assign to the most frequent class (0 after encoding)
                    encoded[i] = 0
            
            X_enc_set[:, idx] = encoded
        
        encoders[idx] = encoder
    
    # Convert entire arrays to float
    X_train_enc = X_train_enc.astype(float)
    X_val_enc = X_val_enc.astype(float)
    X_test_enc = X_test_enc.astype(float)
    
    return X_train_enc, X_val_enc, X_test_enc, encoders


def preprocess_for_decision_trees(X_train, X_val, X_test):
    """
    Preprocess features for Decision Tree models (Classifier and Regressor).
    Decision trees only need categorical encoding, no scaling required.
    
    Args:
        X_train, X_val, X_test (np.ndarray): Raw feature arrays
    
    Returns:
        X_train_prep, X_val_prep, X_test_prep (np.ndarray): Preprocessed features
        preprocessors (dict): Dictionary of fitted preprocessors
    """
    # Only encode categorical features
    X_train_prep, X_val_prep, X_test_prep, encoders = encode_categorical_features(
        X_train, X_val, X_test
    )
    
    preprocessors = {
        'encoders': encoders,
        'scaler': None
    }
    
    return X_train_prep, X_val_prep, X_test_prep, preprocessors


def preprocess_for_linear_models(X_train, X_val, X_test):
    """
    Preprocess features for Linear models (Logistic Regression and Linear Regression).
    Linear models need both categorical encoding and feature scaling.
    
    Args:
        X_train, X_val, X_test (np.ndarray): Raw feature arrays
    
    Returns:
        X_train_prep, X_val_prep, X_test_prep (np.ndarray): Preprocessed features
        preprocessors (dict): Dictionary of fitted preprocessors
    """
    # First encode categorical features
    X_train_enc, X_val_enc, X_test_enc, encoders = encode_categorical_features(
        X_train, X_val, X_test
    )
    
    # Then scale numerical features using StandardScaler
    scaler = StandardScaler()
    
    X_train_prep = X_train_enc.copy()
    X_val_prep = X_val_enc.copy()
    X_test_prep = X_test_enc.copy()
    
    # Scale only numerical features
    X_train_prep[:, NUMERICAL_INDICES] = scaler.fit_transform(
        X_train_enc[:, NUMERICAL_INDICES]
    )
    X_val_prep[:, NUMERICAL_INDICES] = scaler.transform(
        X_val_enc[:, NUMERICAL_INDICES]
    )
    X_test_prep[:, NUMERICAL_INDICES] = scaler.transform(
        X_test_enc[:, NUMERICAL_INDICES]
    )
    
    preprocessors = {
        'encoders': encoders,
        'scaler': scaler
    }
    
    return X_train_prep, X_val_prep, X_test_prep, preprocessors


if __name__ == "__main__":
    from data import load_raw_data, split_data_classification
    
    print("Testing feature preprocessing pipeline...")
    
    X, y, headers = load_raw_data()
    print(f"Loaded data: X shape {X.shape}, y shape {y.shape}")
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_classification(X, y)
    print(f"Split data: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    
    X_train_tree, X_val_tree, X_test_tree, prep_tree = preprocess_for_decision_trees(
        X_train, X_val, X_test
    )
    print(f"Decision Tree preprocessing complete")
    print(f"  Train shape: {X_train_tree.shape}, dtype: {X_train_tree.dtype}")
    print(f"  Val shape: {X_val_tree.shape}, dtype: {X_val_tree.dtype}")
    print(f"  Test shape: {X_test_tree.shape}, dtype: {X_test_tree.dtype}")
    print(f"  Encoders: {len(prep_tree['encoders'])} categorical features encoded")
    print(f"  Scaler: {prep_tree['scaler']}")
    
    X_train_linear, X_val_linear, X_test_linear, prep_linear = preprocess_for_linear_models(
        X_train, X_val, X_test
    )
    print(f"Linear model preprocessing complete")
    print(f"  Train shape: {X_train_linear.shape}, dtype: {X_train_linear.dtype}")
    print(f"  Val shape: {X_val_linear.shape}, dtype: {X_val_linear.dtype}")
    print(f"  Test shape: {X_test_linear.shape}, dtype: {X_test_linear.dtype}")
    print(f"  Encoders: {len(prep_linear['encoders'])} categorical features encoded")
    print(f"  Scaler: {type(prep_linear['scaler']).__name__}")
    
    print(f"Original class distribution: {np.bincount(y_train)}")
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train, y_train)
    print(f"Balanced class distribution: {np.bincount(y_train_balanced)}")
    print(f"Balanced data shape: X {X_train_balanced.shape}, y {y_train_balanced.shape}")
    
    print("All preprocessing tests passed!")
