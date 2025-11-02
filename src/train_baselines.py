"""
Baseline model training for German Credit dataset.
Classification: Logistic Regression, Decision Tree Classifier
Regression: Ridge Regression, Decision Tree Regressor
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data import load_raw_data, split_data_classification, split_data_regression
from features import preprocess_for_linear_models, preprocess_for_decision_trees

RANDOM_SEED = 42


def train_classifiers(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train and evaluate classification models: Logistic Regression and Decision Tree.
    
    Args:
        X_train, X_val, X_test: Feature arrays (raw, not preprocessed)
        y_train, y_val, y_test: Target arrays
    
    Returns:
        results (dict): Dictionary containing trained models and their metrics
    """
    results = {}
    
    print("Training Logistic Regression (Optimized)...")
    X_train_lr, X_val_lr, X_test_lr, prep_lr = preprocess_for_linear_models(
        X_train, X_val, X_test
    )
    
    lr_model = LogisticRegression(
        C=1.0,
        penalty='l1',
        solver='liblinear',
        max_iter=1000,
        random_state=RANDOM_SEED
    )
    lr_model.fit(X_train_lr, y_train)
    
    # Predictions on validation set
    y_val_pred_lr = lr_model.predict(X_val_lr)
    y_val_proba_lr = lr_model.predict_proba(X_val_lr)
    
    # Predictions on test set
    y_test_pred_lr = lr_model.predict(X_test_lr)
    y_test_proba_lr = lr_model.predict_proba(X_test_lr)
    
    # Metrics for both validation and test sets
    results['logistic_regression'] = {
        'model': lr_model,
        'preprocessor': prep_lr,
        'val_accuracy': accuracy_score(y_val, y_val_pred_lr),
        'val_f1_score': f1_score(y_val, y_val_pred_lr, average='weighted'),
        'val_roc_auc': roc_auc_score(y_val, y_val_proba_lr[:, 1]),
        'test_accuracy': accuracy_score(y_test, y_test_pred_lr),
        'test_f1_score': f1_score(y_test, y_test_pred_lr, average='weighted'),
        'test_roc_auc': roc_auc_score(y_test, y_test_proba_lr[:, 1])
    }
    
    print(f"  Val Accuracy:  {results['logistic_regression']['val_accuracy']:.4f}")
    print(f"  Test Accuracy: {results['logistic_regression']['test_accuracy']:.4f}")
    print(f"  Test F1-Score: {results['logistic_regression']['test_f1_score']:.4f}")
    print(f"  Test ROC-AUC:  {results['logistic_regression']['test_roc_auc']:.4f}")
    
    print("\nTraining Decision Tree Classifier (Optimized)...")
    X_train_dt, X_val_dt, X_test_dt, prep_dt = preprocess_for_decision_trees(
        X_train, X_val, X_test
    )
    
    dt_model = DecisionTreeClassifier(
        max_depth=3,
        min_samples_split=100,
        min_samples_leaf=1,
        class_weight=None,
        splitter='random',
        max_features=None,
        ccp_alpha=0.0,
        criterion='gini',
        min_impurity_decrease=0.0,
        random_state=RANDOM_SEED
    )
    dt_model.fit(X_train_dt, y_train)
    
    # Predictions on validation set
    y_val_pred_dt = dt_model.predict(X_val_dt)
    y_val_proba_dt = dt_model.predict_proba(X_val_dt)
    
    # Predictions on test set
    y_test_pred_dt = dt_model.predict(X_test_dt)
    y_test_proba_dt = dt_model.predict_proba(X_test_dt)
    
    # Metrics for both validation and test sets
    results['decision_tree_classifier'] = {
        'model': dt_model,
        'preprocessor': prep_dt,
        'val_accuracy': accuracy_score(y_val, y_val_pred_dt),
        'val_f1_score': f1_score(y_val, y_val_pred_dt, average='weighted'),
        'val_roc_auc': roc_auc_score(y_val, y_val_proba_dt[:, 1]),
        'test_accuracy': accuracy_score(y_test, y_test_pred_dt),
        'test_f1_score': f1_score(y_test, y_test_pred_dt, average='weighted'),
        'test_roc_auc': roc_auc_score(y_test, y_test_proba_dt[:, 1])
    }
    
    print(f"  Val Accuracy:  {results['decision_tree_classifier']['val_accuracy']:.4f}")
    print(f"  Test Accuracy: {results['decision_tree_classifier']['test_accuracy']:.4f}")
    print(f"  Test F1-Score: {results['decision_tree_classifier']['test_f1_score']:.4f}")
    print(f"  Test ROC-AUC:  {results['decision_tree_classifier']['test_roc_auc']:.4f}")
    
    return results


def train_regressors(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train and evaluate regression models: Linear Regression and Decision Tree Regressor.
    
    Args:
        X_train, X_val, X_test: Feature arrays (raw, not preprocessed)
        y_train, y_val, y_test: Target arrays
    
    Returns:
        results (dict): Dictionary containing trained models and their metrics
    """
    results = {}
    
    print("Training Ridge Regression (Optimized)...")
    X_train_lr, X_val_lr, X_test_lr, prep_lr = preprocess_for_linear_models(
        X_train, X_val, X_test
    )
    
    lr_model = Ridge(alpha=10.0, random_state=RANDOM_SEED)
    lr_model.fit(X_train_lr, y_train)
    
    # Predictions on validation set
    y_val_pred_lr = lr_model.predict(X_val_lr)
    
    # Predictions on test set
    y_test_pred_lr = lr_model.predict(X_test_lr)
    
    # Metrics for both validation and test sets
    results['linear_regression'] = {
        'model': lr_model,
        'preprocessor': prep_lr,
        'val_mae': mean_absolute_error(y_val, y_val_pred_lr),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred_lr)),
        'test_mae': mean_absolute_error(y_test, y_test_pred_lr),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred_lr))
    }
    
    print(f"  Val MAE:   {results['linear_regression']['val_mae']:.4f}")
    print(f"  Val RMSE:  {results['linear_regression']['val_rmse']:.4f}")
    print(f"  Test MAE:  {results['linear_regression']['test_mae']:.4f}")
    print(f"  Test RMSE: {results['linear_regression']['test_rmse']:.4f}")
    
    print("\nTraining Decision Tree Regressor (Optimized)...")
    X_train_dt, X_val_dt, X_test_dt, prep_dt = preprocess_for_decision_trees(
        X_train, X_val, X_test
    )
    
    dt_model = DecisionTreeRegressor(
        max_depth=7,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=RANDOM_SEED
    )
    dt_model.fit(X_train_dt, y_train)
    
    # Predictions on validation set
    y_val_pred_dt = dt_model.predict(X_val_dt)
    
    # Predictions on test set
    y_test_pred_dt = dt_model.predict(X_test_dt)
    
    # Metrics for both validation and test sets
    results['decision_tree_regressor'] = {
        'model': dt_model,
        'preprocessor': prep_dt,
        'val_mae': mean_absolute_error(y_val, y_val_pred_dt),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred_dt)),
        'test_mae': mean_absolute_error(y_test, y_test_pred_dt),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred_dt))
    }
    
    print(f"  Val MAE:   {results['decision_tree_regressor']['val_mae']:.4f}")
    print(f"  Val RMSE:  {results['decision_tree_regressor']['val_rmse']:.4f}")
    print(f"  Test MAE:  {results['decision_tree_regressor']['test_mae']:.4f}")
    print(f"  Test RMSE: {results['decision_tree_regressor']['test_rmse']:.4f}")
    
    return results


if __name__ == "__main__":    
    X, y, headers = load_raw_data()
    print(f"\nLoaded data: {X.shape[0]} samples, {X.shape[1]} features")
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_classification(X, y)
    print(f"Split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    classification_results = train_classifiers(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    # Save classification results to CSV
    classification_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Decision Tree Classifier'],
        'Val_Accuracy': [
            classification_results['logistic_regression']['val_accuracy'],
            classification_results['decision_tree_classifier']['val_accuracy']
        ],
        'Val_F1-Score': [
            classification_results['logistic_regression']['val_f1_score'],
            classification_results['decision_tree_classifier']['val_f1_score']
        ],
        'Val_ROC-AUC': [
            classification_results['logistic_regression']['val_roc_auc'],
            classification_results['decision_tree_classifier']['val_roc_auc']
        ],
        'Test_Accuracy': [
            classification_results['logistic_regression']['test_accuracy'],
            classification_results['decision_tree_classifier']['test_accuracy']
        ],
        'Test_F1-Score': [
            classification_results['logistic_regression']['test_f1_score'],
            classification_results['decision_tree_classifier']['test_f1_score']
        ],
        'Test_ROC-AUC': [
            classification_results['logistic_regression']['test_roc_auc'],
            classification_results['decision_tree_classifier']['test_roc_auc']
        ]
    })
    classification_df.to_csv('reports/tables_midpointsub/classification_metrics.csv', index=False)
    print(f"\nClassification metrics saved to reports/tables_midpointsub/classification_metrics.csv")
    

    X_train, X_val, X_test, y_train, y_val, y_test = split_data_regression(X, y)
    print(f"Split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    regression_results = train_regressors(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    # Save regression results to CSV
    regression_df = pd.DataFrame({
        'Model': ['Ridge Regression', 'Decision Tree Regressor'],
        'Val_MAE': [
            regression_results['linear_regression']['val_mae'],
            regression_results['decision_tree_regressor']['val_mae']
        ],
        'Val_RMSE': [
            regression_results['linear_regression']['val_rmse'],
            regression_results['decision_tree_regressor']['val_rmse']
        ],
        'Test_MAE': [
            regression_results['linear_regression']['test_mae'],
            regression_results['decision_tree_regressor']['test_mae']
        ],
        'Test_RMSE': [
            regression_results['linear_regression']['test_rmse'],
            regression_results['decision_tree_regressor']['test_rmse']
        ]
    })
    regression_df.to_csv('reports/tables_midpointsub/regression_metrics.csv', index=False)
    print(f"\nRegression metrics saved to reports/tables_midpointsub/regression_metrics.csv")