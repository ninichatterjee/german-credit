"""
Hyperparameter tuning for Neural Networks (MLP).
Tunes: hidden_layer_sizes, learning_rate, alpha (L2 regularization), batch_size
"""
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, precision_score, recall_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE

from data import load_raw_data, split_data_classification, split_data_regression
from features import preprocess_for_linear_models
from config import RANDOM_SEED

# Centralized hyperparameter grid for both classification and regression
# FAST TUNING: Reduced grid for quick results (36 combinations)
HIDDEN_LAYER_OPTIONS = [
    # (16,),           # Single layer - very small
    # (32,),           # Single layer - small
    # (64,),           # Single layer - medium
    # (128,),          # Single layer - large
    # (256,),          # Single layer - very large
    # (16, 8),         # Two layers - very small
    # (32, 16),        # Two layers - small
    # (64, 32),        # Two layers - medium (baseline)
    # (128, 64),       # Two layers - large
    # (256, 128),      # Two layers - very large
    # (32, 16, 8),     # Three layers - small
    (64, 32, 16),    # Three layers - medium (FIXED FOR TUNING)
    # (128, 64, 32),   # Three layers - large (COMMENTED FOR SPEED)
]

LEARNING_RATE_OPTIONS = [round(0.004 + i * 0.001, 4) for i in range(5)]  # FULL: 0.004 to 0.008 (5 values)

ALPHA_OPTIONS = [round(0.0003 + i * 0.0001, 4) for i in range(5)]  # FULL: 0.0003 to 0.0007 (5 values)

BATCH_SIZE_OPTIONS = [10, 11, 12, 13] 
# BATCH_SIZE_OPTIONS = [6, 7, 8, 9, 10]  # FULL: 5 values
# BATCH_SIZE_OPTIONS = [8, 16, 32, 64, 128]  # 5 values


def tune_mlp_classifier(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Tune MLP Classifier hyperparameters.
    Tests different architectures, learning rates, and regularization.
    
    Returns:
        best_params (dict): Best hyperparameter combination
        results_df (DataFrame): All results for analysis
        best_model: Trained model with best parameters
    """
    print("=== Tuning MLP Classifier ===\n")
    
    # Preprocess data
    X_train_prep, X_val_prep, X_test_prep, preprocessor = preprocess_for_linear_models(
        X_train, X_val, X_test
    )
    
    # Apply SMOTE to training data
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_prep, y_train)
    print(f"Training samples after SMOTE: {len(y_train_balanced)} (from {len(y_train)})\n")
    
    results = []
    total_combinations = (
        len(HIDDEN_LAYER_OPTIONS) * 
        len(LEARNING_RATE_OPTIONS) * 
        len(ALPHA_OPTIONS) * 
        len(BATCH_SIZE_OPTIONS)
    )
    
    print(f"Testing {total_combinations} hyperparameter combinations...\n")
    
    iteration = 0
    for hidden_layers in HIDDEN_LAYER_OPTIONS:
        for lr in LEARNING_RATE_OPTIONS:
            for alpha in ALPHA_OPTIONS:
                for batch_size in BATCH_SIZE_OPTIONS:
                    iteration += 1
                    
                    # Train model
                    mlp = MLPClassifier(
                        hidden_layer_sizes=hidden_layers,
                        activation='relu',
                        solver='adam',
                        alpha=alpha,
                        batch_size=batch_size,
                        learning_rate_init=lr,
                        max_iter=1000,
                        random_state=RANDOM_SEED,
                        early_stopping=True,
                        validation_fraction=0.2,
                        n_iter_no_change=10,
                        shuffle=True,
                        verbose=False
                    )
                    
                    mlp.fit(X_train_balanced, y_train_balanced)
                    
                    # Evaluate on validation set
                    y_val_pred = mlp.predict(X_val_prep)
                    y_val_proba = mlp.predict_proba(X_val_prep)[:, 1]
                    
                    val_accuracy = accuracy_score(y_val, y_val_pred)
                    val_precision = precision_score(y_val, y_val_pred, zero_division=0)
                    val_recall = recall_score(y_val, y_val_pred, zero_division=0)
                    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
                    val_roc_auc = roc_auc_score(y_val, y_val_proba)
                    
                    # Evaluate on test set
                    y_test_pred = mlp.predict(X_test_prep)
                    y_test_proba = mlp.predict_proba(X_test_prep)[:, 1]
                    
                    test_accuracy = accuracy_score(y_test, y_test_pred)
                    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
                    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
                    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
                    test_roc_auc = roc_auc_score(y_test, y_test_proba)
                    
                    results.append({
                        'hidden_layers': str(hidden_layers),
                        'n_layers': len(hidden_layers),
                        'total_units': sum(hidden_layers),
                        'learning_rate': lr,
                        'alpha': alpha,
                        'batch_size': batch_size,
                        'n_iterations': mlp.n_iter_,
                        'val_accuracy': val_accuracy,
                        'val_precision': val_precision,
                        'val_recall': val_recall,
                        'val_f1_score': val_f1,
                        'val_roc_auc': val_roc_auc,
                        'test_accuracy': test_accuracy,
                        'test_f1_score': test_f1,
                        'test_roc_auc': test_roc_auc
                    })
                    
                    if iteration % 20 == 0:
                        print(f"Progress: {iteration}/{total_combinations} combinations tested")
    
    results_df = pd.DataFrame(results)
    
    # Find best parameters based on validation F1 score
    best_idx = results_df['val_f1_score'].idxmax()
    best_params = results_df.loc[best_idx].to_dict()
    
    print(f"\n=== Best Hyperparameters (by validation F1) ===")
    print(f"Hidden layers: {best_params['hidden_layers']}")
    print(f"Learning rate: {best_params['learning_rate']}")
    print(f"Alpha (L2): {best_params['alpha']}")
    print(f"Batch size: {best_params['batch_size']}")
    print(f"Converged in: {int(best_params['n_iterations'])} iterations")
    print(f"\nValidation metrics:")
    print(f"  Accuracy: {best_params['val_accuracy']:.4f}")
    print(f"  F1-Score: {best_params['val_f1_score']:.4f}")
    print(f"  ROC-AUC: {best_params['val_roc_auc']:.4f}")
    print(f"\nTest metrics:")
    print(f"  Accuracy: {best_params['test_accuracy']:.4f}")
    print(f"  F1-Score: {best_params['test_f1_score']:.4f}")
    print(f"  ROC-AUC: {best_params['test_roc_auc']:.4f}")
    
    # Train final model with best parameters
    best_hidden_layers = eval(best_params['hidden_layers'])
    best_model = MLPClassifier(
        hidden_layer_sizes=best_hidden_layers,
        activation='relu',
        solver='adam',
        alpha=best_params['alpha'],
        batch_size=int(best_params['batch_size']),
        learning_rate_init=best_params['learning_rate'],
        max_iter=1000,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        shuffle=True,
        verbose=False
    )
    best_model.fit(X_train_prep, y_train)
    
    return best_params, results_df, best_model


def tune_mlp_regressor(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Tune MLP Regressor hyperparameters.
    Tests different architectures, learning rates, and regularization.
    
    Returns:
        best_params (dict): Best hyperparameter combination
        results_df (DataFrame): All results for analysis
        best_model: Trained model with best parameters
    """
    print("=== Tuning MLP Regressor ===\n")
    
    # Preprocess data
    X_train_prep, X_val_prep, X_test_prep, preprocessor = preprocess_for_linear_models(
        X_train, X_val, X_test
    )
    
    results = []
    total_combinations = (
        len(HIDDEN_LAYER_OPTIONS) * 
        len(LEARNING_RATE_OPTIONS) * 
        len(ALPHA_OPTIONS) * 
        len(BATCH_SIZE_OPTIONS)
    )
    
    print(f"Testing {total_combinations} hyperparameter combinations...\n")
    
    iteration = 0
    for hidden_layers in HIDDEN_LAYER_OPTIONS:
        for lr in LEARNING_RATE_OPTIONS:
            for alpha in ALPHA_OPTIONS:
                for batch_size in BATCH_SIZE_OPTIONS:
                    iteration += 1
                    
                    # Train model
                    mlp = MLPRegressor(
                        hidden_layer_sizes=hidden_layers,
                        activation='relu',
                        solver='adam',
                        alpha=alpha,
                        batch_size=batch_size,
                        learning_rate_init=lr,
                        max_iter=1000,
                        random_state=RANDOM_SEED,
                        early_stopping=True,
                        validation_fraction=0.2,
                        n_iter_no_change=10,
                        shuffle=True,
                        verbose=False
                    )
                    
                    mlp.fit(X_train_prep, y_train)
                    
                    # Evaluate on validation set
                    y_val_pred = mlp.predict(X_val_prep)
                    val_mae = mean_absolute_error(y_val, y_val_pred)
                    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                    
                    # Evaluate on test set
                    y_test_pred = mlp.predict(X_test_prep)
                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                    
                    results.append({
                        'hidden_layers': str(hidden_layers),
                        'n_layers': len(hidden_layers),
                        'total_units': sum(hidden_layers),
                        'learning_rate': lr,
                        'alpha': alpha,
                        'batch_size': batch_size,
                        'n_iterations': mlp.n_iter_,
                        'val_mae': val_mae,
                        'val_rmse': val_rmse,
                        'test_mae': test_mae,
                        'test_rmse': test_rmse
                    })
                    
                    if iteration % 20 == 0:
                        print(f"Progress: {iteration}/{total_combinations} combinations tested")
    
    results_df = pd.DataFrame(results)
    
    # Find best parameters based on validation MAE (lower is better)
    best_idx = results_df['val_mae'].idxmin()
    best_params = results_df.loc[best_idx].to_dict()
    
    print(f"\n=== Best Hyperparameters (by validation MAE) ===")
    print(f"Hidden layers: {best_params['hidden_layers']}")
    print(f"Learning rate: {best_params['learning_rate']}")
    print(f"Alpha (L2): {best_params['alpha']}")
    print(f"Batch size: {best_params['batch_size']}")
    print(f"Converged in: {int(best_params['n_iterations'])} iterations")
    print(f"\nValidation metrics:")
    print(f"  MAE: {best_params['val_mae']:.4f}")
    print(f"  RMSE: {best_params['val_rmse']:.4f}")
    print(f"\nTest metrics:")
    print(f"  MAE: {best_params['test_mae']:.4f}")
    print(f"  RMSE: {best_params['test_rmse']:.4f}")
    
    # Train final model with best parameters
    best_hidden_layers = eval(best_params['hidden_layers'])
    best_model = MLPRegressor(
        hidden_layer_sizes=best_hidden_layers,
        activation='relu',
        solver='adam',
        alpha=best_params['alpha'],
        batch_size=int(best_params['batch_size']),
        learning_rate_init=best_params['learning_rate'],
        max_iter=1000,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        shuffle=True,
        verbose=False
    )
    best_model.fit(X_train_prep, y_train)
    
    return best_params, results_df, best_model


def analyze_hyperparameter_impact_classifier(results_df):
    """Analyze how different hyperparameters affect classification performance."""
    print("\n" + "="*60)
    print("HYPERPARAMETER IMPACT ANALYSIS - CLASSIFICATION")
    print("="*60)
    
    print("\n--- Impact of Architecture Depth (Number of Layers) ---")
    depth_impact = results_df.groupby('n_layers')['val_f1_score'].agg(['mean', 'std', 'max'])
    print(depth_impact.to_string())
    
    print("\n--- Impact of Network Size (Total Units) ---")
    size_bins = pd.cut(results_df['total_units'], bins=[0, 50, 100, 150, 300])
    size_impact = results_df.groupby(size_bins)['val_f1_score'].agg(['mean', 'std', 'max'])
    print(size_impact.to_string())
    
    print("\n--- Impact of Learning Rate ---")
    lr_impact = results_df.groupby('learning_rate')['val_f1_score'].agg(['mean', 'std', 'max'])
    print(lr_impact.to_string())
    
    print("\n--- Impact of Alpha (L2 Regularization) ---")
    alpha_impact = results_df.groupby('alpha')['val_f1_score'].agg(['mean', 'std', 'max'])
    print(alpha_impact.to_string())
    
    print("\n--- Impact of Batch Size ---")
    batch_impact = results_df.groupby('batch_size')['val_f1_score'].agg(['mean', 'std', 'max'])
    print(batch_impact.to_string())
    
    print("\n--- Top 10 Architectures ---")
    top_10 = results_df.nlargest(10, 'val_f1_score')[
        ['hidden_layers', 'learning_rate', 'alpha', 'batch_size', 'n_iterations', 
         'val_accuracy', 'val_f1_score', 'val_roc_auc']
    ]
    print(top_10.to_string(index=False))


def analyze_hyperparameter_impact_regressor(results_df):
    """Analyze how different hyperparameters affect regression performance."""
    print("\n" + "="*60)
    print("HYPERPARAMETER IMPACT ANALYSIS - REGRESSION")
    print("="*60)
    
    print("\n--- Impact of Architecture Depth (Number of Layers) ---")
    depth_impact = results_df.groupby('n_layers')['val_mae'].agg(['mean', 'std', 'min'])
    print(depth_impact.to_string())
    
    print("\n--- Impact of Network Size (Total Units) ---")
    size_bins = pd.cut(results_df['total_units'], bins=[0, 50, 100, 150, 300])
    size_impact = results_df.groupby(size_bins)['val_mae'].agg(['mean', 'std', 'min'])
    print(size_impact.to_string())
    
    print("\n--- Impact of Learning Rate ---")
    lr_impact = results_df.groupby('learning_rate')['val_mae'].agg(['mean', 'std', 'min'])
    print(lr_impact.to_string())
    
    print("\n--- Impact of Alpha (L2 Regularization) ---")
    alpha_impact = results_df.groupby('alpha')['val_mae'].agg(['mean', 'std', 'min'])
    print(alpha_impact.to_string())
    
    print("\n--- Impact of Batch Size ---")
    batch_impact = results_df.groupby('batch_size')['val_mae'].agg(['mean', 'std', 'min'])
    print(batch_impact.to_string())
    
    print("\n--- Top 10 Architectures ---")
    top_10 = results_df.nsmallest(10, 'val_mae')[
        ['hidden_layers', 'learning_rate', 'alpha', 'batch_size', 'n_iterations', 
         'val_mae', 'val_rmse']
    ]
    print(top_10.to_string(index=False))


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    run_classification = True
    run_regression = True
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--classification' or sys.argv[1] == '--clf':
            run_regression = False
        elif sys.argv[1] == '--regression' or sys.argv[1] == '--reg':
            run_classification = False
    
    print("="*60)
    print("NEURAL NETWORK HYPERPARAMETER TUNING")
    print("="*60)
    
    X, y, headers = load_raw_data()
    print(f"\nLoaded data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Tune classification MLP
    if run_classification:
        print("\n" + "="*60)
        print("CLASSIFICATION TASK")
        print("="*60)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data_classification(X, y)
        print(f"Split data: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}\n")
        
        best_params_clf, results_clf, best_model_clf = tune_mlp_classifier(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        analyze_hyperparameter_impact_classifier(results_clf)
        
        # Save results
        results_clf.to_csv('reports/mlp_classifier_tuning_results.csv', index=False)
        print("\nSaved: reports/mlp_classifier_tuning_results.csv")
    
    # Tune regression MLP
    if run_regression:
        print("\n" + "="*60)
        print("REGRESSION TASK")
        print("="*60)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data_regression(X, y)
        print(f"Split data: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}\n")
        
        best_params_reg, results_reg, best_model_reg = tune_mlp_regressor(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        analyze_hyperparameter_impact_regressor(results_reg)
        
        # Save results
        results_reg.to_csv('reports/mlp_regressor_tuning_results.csv', index=False)
        print("\nSaved: reports/mlp_regressor_tuning_results.csv")
    
    print("\n" + "="*60)
    print("TUNING COMPLETE")
    print("="*60)
