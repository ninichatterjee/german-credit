import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error
)
from imblearn.over_sampling import SMOTE
import pickle
import warnings

# Try to import TensorFlow, fall back to sklearn if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    import mlflow.tensorflow
    TF_AVAILABLE = True
    print("TensorFlow available - using advanced NN with dropout and batch normalization")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available - using sklearn MLPClassifier")
    warnings.warn("Install tensorflow for better NN performance: pip install tensorflow")

from data import load_raw_data, split_data_classification, split_data_regression
from features import preprocess_for_linear_models
from config import (
    RANDOM_SEED, MLFLOW_EXPERIMENT_NN, NN_HIDDEN_LAYERS, NN_ACTIVATION,
    NN_SOLVER, NN_LEARNING_RATE, NN_BATCH_SIZE, NN_ALPHA, NN_MAX_ITER,
    NN_EARLY_STOPPING, NN_VALIDATION_FRACTION, NN_PATIENCE, PLOTS_FINAL_DIR
)
from utils import set_random_seeds, print_class_distribution, print_model_summary

set_random_seeds(RANDOM_SEED)

mlflow.set_experiment(MLFLOW_EXPERIMENT_NN)

def create_tf_classification_model(input_dim, dropout_rate=0.5, l2_lambda=0.001):
    """
    Create TensorFlow model with improved regularization.
    Architecture: Input -> 64 (BN, Dropout, L2) -> 32 (BN, Dropout, L2) -> 1 (sigmoid)
    
    Args:
        input_dim: Number of input features
        dropout_rate: Dropout rate (default: 0.5)
        l2_lambda: L2 regularization factor (default: 0.001)
    """
    model = keras.Sequential([
        # First hidden layer
        layers.Dense(64, input_dim=input_dim, 
                    kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),
        
        # Second hidden layer
        layers.Dense(32, kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),
        
        # Output layer
        layers.Dense(1, activation='sigmoid', 
                    kernel_regularizer=regularizers.l2(l2_lambda))
    ])
    
    # Use a lower learning rate with Adam optimizer
    optimizer = keras.optimizers.Adam(
        learning_rate=0.0005,  # Reduced from 0.001
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model


def train_classification_nn_tf(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train TensorFlow NN with batch normalization and dropout.
    """
    print("\n=== Training TensorFlow Classification Neural Network ===")
    
    # Preprocess data
    X_train_prep, X_val_prep, X_test_prep, preprocessor = preprocess_for_linear_models(
        X_train, X_val, X_test
    )
    
    # Apply SMOTE
    print_class_distribution(y_train, "BEFORE SMOTE")
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_prep, y_train)
    print_class_distribution(y_train_balanced, "AFTER SMOTE")
    
    # Convert labels to 0/1 for TensorFlow (from 1/2)
    y_train_tf = y_train_balanced - 1
    y_val_tf = y_val - 1
    y_test_tf = y_test - 1
    
    with mlflow.start_run(run_name="TF_MLP_Classification"):
        # Create model with increased dropout and L2 regularization
        model = create_tf_classification_model(
            X_train_balanced.shape[1],
            dropout_rate=0.5,  # Increased from 0.3
            l2_lambda=0.01    # Increased L2 regularization
        )
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,  # Increased patience
            restore_best_weights=True,
            verbose=1
        )
        
        # Learning rate scheduler
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,     # Reduce learning rate by half
            patience=5,     # Wait 5 epochs before reducing LR
            min_lr=1e-6,    # Minimum learning rate
            verbose=1
        )
        
        # Model checkpoint
        checkpoint = keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=0
        )
        
        callbacks = [early_stop, reduce_lr, checkpoint]
        
        # Train model with smaller batch size
        history = model.fit(
            X_train_balanced, y_train_tf,
            validation_data=(X_val_prep, y_val_tf),
            epochs=200,          # Can go up to 200 due to early stopping
            batch_size=16,       # Smaller batch size
            callbacks=callbacks,
            verbose=1,           # Show progress bar
            shuffle=True
        )
        
        # Get predictions
        y_val_proba = model.predict(X_val_prep).flatten()
        y_val_pred = (y_val_proba > 0.5).astype(int) + 1  # Convert back to 1/2
        
        y_test_proba = model.predict(X_test_prep).flatten()
        y_test_pred = (y_test_proba > 0.5).astype(int) + 1  # Convert back to 1/2
        
        # Calculate metrics
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred, pos_label=2, zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, pos_label=2, zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, pos_label=2, zero_division=0)
        val_roc_auc = roc_auc_score(y_val, y_val_proba)
        
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, pos_label=2, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, pos_label=2, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, pos_label=2, zero_division=0)
        test_roc_auc = roc_auc_score(y_test, y_test_proba)
        
        # Log metrics
        mlflow.log_param('framework', 'tensorflow')
        mlflow.log_param('dropout_rate', 0.3)
        mlflow.log_param('batch_normalization', True)
        mlflow.log_param('architecture', str(NN_HIDDEN_LAYERS))
        
        mlflow.log_metric('val_accuracy', val_accuracy)
        mlflow.log_metric('val_precision', val_precision)
        mlflow.log_metric('val_recall', val_recall)
        mlflow.log_metric('val_f1_score', val_f1)
        mlflow.log_metric('val_roc_auc', val_roc_auc)
        
        mlflow.log_metric('test_accuracy', test_accuracy)
        mlflow.log_metric('test_precision', test_precision)
        mlflow.log_metric('test_recall', test_recall)
        mlflow.log_metric('test_f1_score', test_f1)
        mlflow.log_metric('test_roc_auc', test_roc_auc)
        
        # Save model
        mlflow.tensorflow.log_model(model, "model")
        
        # Save loss curves
        with open(f'{PLOTS_FINAL_DIR}/classification_nn_loss_curve.pkl', 'wb') as f:
            pickle.dump(history.history, f)
        
        print(f"  TensorFlow Model Performance:")
        print(f"  Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val ROC-AUC: {val_roc_auc:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}, Test ROC-AUC: {test_roc_auc:.4f}")
        
        return {
            'val_accuracy': val_accuracy,
            'val_f1': val_f1,
            'val_roc_auc': val_roc_auc,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'test_roc_auc': test_roc_auc,
            'history': history.history
        }


def train_classification_nn(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train MLP classifier - uses TensorFlow if available, otherwise sklearn.
    Architecture: Input -> 64 -> 32 -> 1 (sigmoid)
    """
    if TF_AVAILABLE:
        return train_classification_nn_tf(X_train, X_val, X_test, y_train, y_val, y_test)
    print("\n=== Training Classification Neural Network ===")
    
    # Preprocess data (same as linear models: OneHot + Scaling)
    X_train_prep, X_val_prep, X_test_prep, preprocessor = preprocess_for_linear_models(
        X_train, X_val, X_test
    )
    
    # Apply SMOTE to handle class imbalance (70% good, 30% bad)
    print_class_distribution(y_train, "BEFORE SMOTE")
    
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_prep, y_train)
    
    print_class_distribution(y_train_balanced, "AFTER SMOTE")
    print(f"  Total samples: {len(y_train_balanced)} (increased from {len(y_train)})")
    
    with mlflow.start_run(run_name="MLP_Classification"):
        # Define MLP architecture with class weights
        mlp = MLPClassifier(
            hidden_layer_sizes=NN_HIDDEN_LAYERS,
            activation=NN_ACTIVATION,
            solver=NN_SOLVER,
            alpha=NN_ALPHA,
            batch_size=NN_BATCH_SIZE,
            learning_rate_init=NN_LEARNING_RATE,
            max_iter=NN_MAX_ITER,
            random_state=RANDOM_SEED,
            early_stopping=NN_EARLY_STOPPING,
            validation_fraction=NN_VALIDATION_FRACTION,
            n_iter_no_change=NN_PATIENCE,
            shuffle=True,
            verbose=False
        )
        
        # Log parameters
        mlflow.log_param('model_type', 'classification')
        mlflow.log_param('architecture', f'{NN_HIDDEN_LAYERS[0]}-{NN_HIDDEN_LAYERS[1]}-{NN_HIDDEN_LAYERS[2] if len(NN_HIDDEN_LAYERS) > 2 else 1}')
        mlflow.log_param('activation', NN_ACTIVATION)
        mlflow.log_param('solver', NN_SOLVER)
        mlflow.log_param('learning_rate', NN_LEARNING_RATE)
        mlflow.log_param('batch_size', NN_BATCH_SIZE)
        mlflow.log_param('alpha', NN_ALPHA)
        mlflow.log_param('max_iter', NN_MAX_ITER)
        mlflow.log_param('early_stopping', NN_EARLY_STOPPING)
        mlflow.log_param('patience', NN_PATIENCE)
        mlflow.log_param('preprocessing', 'one_hot_encoding + scaling + SMOTE')
        mlflow.log_param('smote_applied', True)
        mlflow.log_param('samples_after_smote', len(y_train_balanced))
        
        # Train with SMOTE-balanced data
        mlp.fit(X_train_balanced, y_train_balanced)
        
        # Get predictions
        y_val_pred = mlp.predict(X_val_prep)
        y_val_proba = mlp.predict_proba(X_val_prep)[:, 1]
        y_test_pred = mlp.predict(X_test_prep)
        y_test_proba = mlp.predict_proba(X_test_prep)[:, 1]
        
        # Calculate metrics (pos_label=2 for Bad Credit class)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred, pos_label=2, zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, pos_label=2, zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, pos_label=2, zero_division=0)
        val_roc_auc = roc_auc_score(y_val, y_val_proba)
        
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, pos_label=2, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, pos_label=2, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, pos_label=2, zero_division=0)
        test_roc_auc = roc_auc_score(y_test, y_test_proba)
        
        # Log metrics
        mlflow.log_metric('val_accuracy', val_accuracy)
        mlflow.log_metric('val_precision', val_precision)
        mlflow.log_metric('val_recall', val_recall)
        mlflow.log_metric('val_f1_score', val_f1)
        mlflow.log_metric('val_roc_auc', val_roc_auc)
        
        mlflow.log_metric('test_accuracy', test_accuracy)
        mlflow.log_metric('test_precision', test_precision)
        mlflow.log_metric('test_recall', test_recall)
        mlflow.log_metric('test_f1_score', test_f1)
        mlflow.log_metric('test_roc_auc', test_roc_auc)
        
        # Log training info
        mlflow.log_metric('n_iterations', mlp.n_iter_)
        mlflow.log_metric('n_layers', mlp.n_layers_)
        
        # Save model to MLflow
        mlflow.sklearn.log_model(mlp, "model")
        
        # Save loss curve for plotting
        loss_curve = mlp.loss_curve_
        with open(f'{PLOTS_FINAL_DIR}/classification_nn_loss_curve.pkl', 'wb') as f:
            pickle.dump(loss_curve, f)
        
        print(f"  Model saved to MLflow")
        print(f"  Converged after {mlp.n_iter_} iterations")
        print(f"  Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val ROC-AUC: {val_roc_auc:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}, Test ROC-AUC: {test_roc_auc:.4f}")
        
        results = {
            'model': mlp,
            'preprocessor': preprocessor,
            'loss_curve': loss_curve,
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1_score': val_f1,
            'val_roc_auc': val_roc_auc,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1_score': test_f1,
            'test_roc_auc': test_roc_auc,
            'y_test_pred': y_test_pred,
            'y_test_proba': y_test_proba
        }
        
        return results


def train_regression_nn(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train MLP regressor with early stopping and loss curve tracking.
    Architecture: Input -> 64 -> 32 -> 1 (linear)
    """
    print("\n=== Training Regression Neural Network ===")
    
    # Preprocess data
    X_train_prep, X_val_prep, X_test_prep, preprocessor = preprocess_for_linear_models(
        X_train, X_val, X_test
    )
    
    with mlflow.start_run(run_name="MLP_Regression"):
        # Define MLP architecture
        mlp = MLPRegressor(
            hidden_layer_sizes=NN_HIDDEN_LAYERS,
            activation=NN_ACTIVATION,
            solver=NN_SOLVER,
            alpha=NN_ALPHA,
            batch_size=NN_BATCH_SIZE,
            learning_rate_init=NN_LEARNING_RATE,
            max_iter=NN_MAX_ITER,
            random_state=RANDOM_SEED,
            early_stopping=NN_EARLY_STOPPING,
            validation_fraction=NN_VALIDATION_FRACTION,
            n_iter_no_change=NN_PATIENCE,
            shuffle=True,
            verbose=False
        )
        
        # Log parameters
        mlflow.log_param('model_type', 'regression')
        mlflow.log_param('architecture', f'{NN_HIDDEN_LAYERS[0]}-{NN_HIDDEN_LAYERS[1]}-1')
        mlflow.log_param('activation', NN_ACTIVATION)
        mlflow.log_param('solver', NN_SOLVER)
        mlflow.log_param('learning_rate', NN_LEARNING_RATE)
        mlflow.log_param('batch_size', NN_BATCH_SIZE)
        mlflow.log_param('alpha', NN_ALPHA)
        mlflow.log_param('max_iter', NN_MAX_ITER)
        mlflow.log_param('early_stopping', NN_EARLY_STOPPING)
        mlflow.log_param('patience', NN_PATIENCE)
        mlflow.log_param('preprocessing', 'one_hot_encoding + scaling')
        
        # Train
        mlp.fit(X_train_prep, y_train)
        
        # Get predictions
        y_val_pred = mlp.predict(X_val_prep)
        y_test_pred = mlp.predict(X_test_prep)
        
        # Calculate metrics
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Log metrics
        mlflow.log_metric('val_mae', val_mae)
        mlflow.log_metric('val_rmse', val_rmse)
        mlflow.log_metric('test_mae', test_mae)
        mlflow.log_metric('test_rmse', test_rmse)
        
        # Log training info
        mlflow.log_metric('n_iterations', mlp.n_iter_)
        mlflow.log_metric('n_layers', mlp.n_layers_)
        
        # Save model
        mlflow.sklearn.log_model(mlp, "model")
        
        # Save loss curve for plotting
        loss_curve = mlp.loss_curve_
        with open(f'{PLOTS_FINAL_DIR}/regression_nn_loss_curve.pkl', 'wb') as f:
            pickle.dump(loss_curve, f)
        
        print(f"  Converged after {mlp.n_iter_} iterations")
        print(f"  Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}")
        print(f"  Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}")
        
        results = {
            'model': mlp,
            'preprocessor': preprocessor,
            'loss_curve': loss_curve,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'y_test_pred': y_test_pred
        }
        
        return results


if __name__ == "__main__":
    # Load data
    X, y, headers = load_raw_data()
    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features\n")
    
    # Train classification NN
    print("=" * 60)
    print("CLASSIFICATION TASK")
    print("=" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_classification(X, y)
    classification_results = train_classification_nn(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Train regression NN
    print("\n" + "=" * 60)
    print("REGRESSION TASK")
    print("=" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_regression(X, y)
    regression_results = train_regression_nn(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\nResults saved:")
    print("  - MLflow logs: mlruns/")
    print(f"  - Loss curves: {PLOTS_FINAL_DIR}/")