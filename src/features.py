"""
Feature preprocessing and preprocessing for the models:
Logistic Regression, Linear Regression, Decision Tree Classifier, Decision Tree Regressor
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

from config import RANDOM_SEED, CATEGORICAL_INDICES, NUMERICAL_INDICES

def encode_categorical_features(X_train, X_val, X_test):
    """
    Encode categorical features using LabelEncoder.
    Fits on training data and transforms all sets.
    """
    X_train_enc = X_train.copy()
    X_val_enc = X_val.copy()
    X_test_enc = X_test.copy()
    
    encoders = {}
    
    for idx in NUMERICAL_INDICES:
        X_train_enc[:, idx] = X_train_enc[:, idx].astype(float)
        X_val_enc[:, idx] = X_val_enc[:, idx].astype(float)
        X_test_enc[:, idx] = X_test_enc[:, idx].astype(float)
    
    for idx in CATEGORICAL_INDICES:
        encoder = LabelEncoder()
        
        X_train_enc[:, idx] = encoder.fit_transform(X_train[:, idx])
        
        for X_set, X_enc_set in [(X_val, X_val_enc), (X_test, X_test_enc)]:
            labels = X_set[:, idx]
            encoded = np.zeros(len(labels), dtype=int)
            
            for i, label in enumerate(labels):
                if label in encoder.classes_:
                    encoded[i] = encoder.transform([label])[0]
                else:
                    encoded[i] = 0
            
            X_enc_set[:, idx] = encoded
        
        encoders[idx] = encoder
    
    X_train_enc = X_train_enc.astype(float)
    X_val_enc = X_val_enc.astype(float)
    X_test_enc = X_test_enc.astype(float)
    
    return X_train_enc, X_val_enc, X_test_enc, encoders


def preprocess_for_decision_trees(X_train, X_val, X_test):
    """
    Preprocess features for Decision Tree models (Classifier and Regressor).
    Decision trees only need categorical encoding, no scaling required.
    """
    X_train_prep, X_val_prep, X_test_prep, encoders = encode_categorical_features(
        X_train, X_val, X_test
    )
    
    preprocessors = {
        'encoders': encoders,
        'scaler': None
    }
    
    return X_train_prep, X_val_prep, X_test_prep, preprocessors


def preprocess_for_linear_models(X_train, X_val, X_test, add_interactions=False):
    """
    Preprocess features for Linear models (Logistic Regression and Linear Regression).
    Uses one-hot encoding for categorical features and scaling for numerical features.
    
    Args:
        X_train, X_val, X_test: Raw feature arrays
        add_interactions: If True, adds interaction features (e.g., feature1 * feature2)
    """
    X_train_num = X_train[:, NUMERICAL_INDICES].astype(float)
    X_val_num = X_val[:, NUMERICAL_INDICES].astype(float)
    X_test_num = X_test[:, NUMERICAL_INDICES].astype(float)
    
    X_train_cat = X_train[:, CATEGORICAL_INDICES]
    X_val_cat = X_val[:, CATEGORICAL_INDICES]
    X_test_cat = X_test[:, CATEGORICAL_INDICES]
    
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    X_train_cat_encoded = onehot_encoder.fit_transform(X_train_cat)
    X_val_cat_encoded = onehot_encoder.transform(X_val_cat)
    X_test_cat_encoded = onehot_encoder.transform(X_test_cat)
    
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_val_num_scaled = scaler.transform(X_val_num)
    X_test_num_scaled = scaler.transform(X_test_num)
    
    X_train_prep = np.hstack([X_train_num_scaled, X_train_cat_encoded])
    X_val_prep = np.hstack([X_val_num_scaled, X_val_cat_encoded])
    X_test_prep = np.hstack([X_test_num_scaled, X_test_cat_encoded])
    
    if add_interactions:
        train_interaction = X_train_num[:, 0] * X_train_num[:, 2]  # Duration × Credit
        val_interaction = X_val_num[:, 0] * X_val_num[:, 2]
        test_interaction = X_test_num[:, 0] * X_test_num[:, 2]
        
        X_train_prep = np.hstack([X_train_prep, train_interaction.reshape(-1, 1)])
        X_val_prep = np.hstack([X_val_prep, val_interaction.reshape(-1, 1)])
        X_test_prep = np.hstack([X_test_prep, test_interaction.reshape(-1, 1)])
    
    preprocessors = {
        'onehot_encoder': onehot_encoder,
        'scaler': scaler,
        'interactions': add_interactions
    }
    
    return X_train_prep, X_val_prep, X_test_prep, preprocessors


if __name__ == "__main__":
    from data import load_raw_data, split_data_classification
    
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
    
    print("All preprocessing tests passed!")
