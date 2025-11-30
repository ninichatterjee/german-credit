"""
Baseline model training with MLflow tracking.
Classification Models: Logistic Regression, Decision Tree Classifier
Regression Models: Linear Regression, Decision Tree Regressor
"""
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_squared_error, mean_absolute_error
)

from data import load_raw_data, split_data_classification, split_data_regression
from features import preprocess_for_linear_models, preprocess_for_decision_trees
from config import RANDOM_SEED, MLFLOW_EXPERIMENT_BASELINE
from utils import set_random_seeds, print_model_summary, calculate_class_weights

set_random_seeds(RANDOM_SEED)
mlflow.set_experiment(MLFLOW_EXPERIMENT_BASELINE)

def train_classifiers(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train and evaluate classification models: Logistic Regression and Decision Tree.
    """
    results = {}

    X_train_lr, X_val_lr, X_test_lr, prep_lr = preprocess_for_linear_models(
        X_train, X_val, X_test
    )
    
    with mlflow.start_run(run_name="Logistic_Regression"):
        params = {
            'C': 1.0,
            'penalty': 'l1',
            'solver': 'liblinear',
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': RANDOM_SEED
        }
        mlflow.log_params(params)
        mlflow.log_param('model_type', 'classification')
        mlflow.log_param('preprocessing', 'one_hot_encoding + scaling')
        
        lr_model = LogisticRegression(**params)
        lr_model.fit(X_train_lr, y_train)
        
        y_val_pred_lr = lr_model.predict(X_val_lr)
        y_val_proba_lr = lr_model.predict_proba(X_val_lr)
        y_test_pred_lr = lr_model.predict(X_test_lr)
        y_test_proba_lr = lr_model.predict_proba(X_test_lr)
        
        val_accuracy = accuracy_score(y_val, y_val_pred_lr)
        val_f1 = f1_score(y_val, y_val_pred_lr, average='weighted')
        val_roc_auc = roc_auc_score(y_val, y_val_proba_lr[:, 1])
        test_accuracy = accuracy_score(y_test, y_test_pred_lr)
        test_f1 = f1_score(y_test, y_test_pred_lr, average='weighted')
        test_roc_auc = roc_auc_score(y_test, y_test_proba_lr[:, 1])
        
        mlflow.log_metric('val_accuracy', val_accuracy)
        mlflow.log_metric('val_f1_score', val_f1)
        mlflow.log_metric('val_roc_auc', val_roc_auc)
        mlflow.log_metric('test_accuracy', test_accuracy)
        mlflow.log_metric('test_f1_score', test_f1)
        mlflow.log_metric('test_roc_auc', test_roc_auc)
        
        mlflow.sklearn.log_model(lr_model, "model")
        
        results['logistic_regression'] = {
            'model': lr_model,
            'preprocessor': prep_lr,
            'val_accuracy': val_accuracy,
            'val_f1_score': val_f1,
            'val_roc_auc': val_roc_auc,
            'test_accuracy': test_accuracy,
            'test_f1_score': test_f1,
            'test_roc_auc': test_roc_auc
        }
    
    print(f"  Test Accuracy: {results['logistic_regression']['test_accuracy']:.4f}")
    
    X_train_dt, X_val_dt, X_test_dt, prep_dt = preprocess_for_decision_trees(
        X_train, X_val, X_test
    )
    
    with mlflow.start_run(run_name="Decision_Tree_Classifier"):
        params = {
            'max_depth': 5,
            'min_samples_split': 100,
            'min_samples_leaf': 2,
            'class_weight': 'balanced',
            'splitter': 'random',
            'max_features': None,
            'ccp_alpha': 0.00025,
            'criterion': 'gini',
            'min_impurity_decrease': 0.0,
            'random_state': RANDOM_SEED
        }
        mlflow.log_params(params)
        mlflow.log_param('model_type', 'classification')
        mlflow.log_param('preprocessing', 'label_encoding')
        
        dt_model = DecisionTreeClassifier(**params)
        dt_model.fit(X_train_dt, y_train)
        
        y_val_pred_dt = dt_model.predict(X_val_dt)
        y_val_proba_dt = dt_model.predict_proba(X_val_dt)
        y_test_pred_dt = dt_model.predict(X_test_dt)
        y_test_proba_dt = dt_model.predict_proba(X_test_dt)
        
        val_accuracy = accuracy_score(y_val, y_val_pred_dt)
        val_f1 = f1_score(y_val, y_val_pred_dt, average='weighted')
        val_roc_auc = roc_auc_score(y_val, y_val_proba_dt[:, 1])
        test_accuracy = accuracy_score(y_test, y_test_pred_dt)
        test_f1 = f1_score(y_test, y_test_pred_dt, average='weighted')
        test_roc_auc = roc_auc_score(y_test, y_test_proba_dt[:, 1])
        
        mlflow.log_metric('val_accuracy', val_accuracy)
        mlflow.log_metric('val_f1_score', val_f1)
        mlflow.log_metric('val_roc_auc', val_roc_auc)
        mlflow.log_metric('test_accuracy', test_accuracy)
        mlflow.log_metric('test_f1_score', test_f1)
        mlflow.log_metric('test_roc_auc', test_roc_auc)
        
        mlflow.sklearn.log_model(dt_model, "model")
        
        results['decision_tree_classifier'] = {
            'model': dt_model,
            'preprocessor': prep_dt,
            'val_accuracy': val_accuracy,
            'val_f1_score': val_f1,
            'val_roc_auc': val_roc_auc,
            'test_accuracy': test_accuracy,
            'test_f1_score': test_f1,
            'test_roc_auc': test_roc_auc
        }
    
    print(f"  Test Accuracy: {results['decision_tree_classifier']['test_accuracy']:.4f}")
    
    return results


def train_regressors(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train and evaluate regression models: Linear Regression and Decision Tree Regressor.
    """
    results = {}

    X_train_lr, X_val_lr, X_test_lr, prep_lr = preprocess_for_linear_models(
        X_train, X_val, X_test
    )
    
    with mlflow.start_run(run_name="Linear_Regression"):
        mlflow.log_param('model_type', 'regression')
        mlflow.log_param('preprocessing', 'one_hot_encoding + scaling')
        mlflow.log_param('fit_intercept', True)
        
        lr_model = LinearRegression()
        lr_model.fit(X_train_lr, y_train)
        
        y_val_pred_lr = lr_model.predict(X_val_lr)
        y_test_pred_lr = lr_model.predict(X_test_lr)
        
        val_mae = mean_absolute_error(y_val, y_val_pred_lr)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_lr))
        test_mae = mean_absolute_error(y_test, y_test_pred_lr)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_lr))
        
        mlflow.log_metric('val_mae', val_mae)
        mlflow.log_metric('val_rmse', val_rmse)
        mlflow.log_metric('test_mae', test_mae)
        mlflow.log_metric('test_rmse', test_rmse)
        
        mlflow.sklearn.log_model(lr_model, "model")
        
        results['linear_regression'] = {
            'model': lr_model,
            'preprocessor': prep_lr,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'test_mae': test_mae,
            'test_rmse': test_rmse
        }
    
    print(f"  Test RMSE: {results['linear_regression']['test_rmse']:.4f}")
    
    X_train_dt, X_val_dt, X_test_dt, prep_dt = preprocess_for_decision_trees(
        X_train, X_val, X_test
    )
    
    with mlflow.start_run(run_name="Decision_Tree_Regressor"):
        params = {
            'max_depth': 5,
            'min_samples_split': 50,
            'min_samples_leaf': 30,
            'random_state': RANDOM_SEED
        }
        mlflow.log_params(params)
        mlflow.log_param('model_type', 'regression')
        mlflow.log_param('preprocessing', 'label_encoding')
        
        dt_model = DecisionTreeRegressor(**params)
        dt_model.fit(X_train_dt, y_train)
        
        y_val_pred_dt = dt_model.predict(X_val_dt)
        y_test_pred_dt = dt_model.predict(X_test_dt)
        
        val_mae = mean_absolute_error(y_val, y_val_pred_dt)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_dt))
        test_mae = mean_absolute_error(y_test, y_test_pred_dt)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_dt))
        
        mlflow.log_metric('val_mae', val_mae)
        mlflow.log_metric('val_rmse', val_rmse)
        mlflow.log_metric('test_mae', test_mae)
        mlflow.log_metric('test_rmse', test_rmse)
        
        mlflow.sklearn.log_model(dt_model, "model")
        
        results['decision_tree_regressor'] = {
            'model': dt_model,
            'preprocessor': prep_dt,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'test_mae': test_mae,
            'test_rmse': test_rmse
        }
    
    print(f"  Test RMSE: {results['decision_tree_regressor']['test_rmse']:.4f}")
    
    return results


if __name__ == "__main__":    
    X, y, headers = load_raw_data()
    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features\n")
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_classification(X, y)
    classification_results = train_classifiers(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
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
    print("Saved: classification_metrics.csv\n")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data_regression(X, y)
    regression_results = train_regressors(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    regression_df = pd.DataFrame({
        'Model': ['Linear Regression', 'Decision Tree Regressor'],
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
    print("Saved: regression_metrics.csv")


if __name__ == "__main__":
    X, y, headers = load_raw_data()
    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    
    print("\n" + "="*60)
    print("CLASSIFICATION TASK")
    print("="*60)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_classification(X, y)
    classification_results = train_classifiers(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    print("\n" + "="*60)
    print("REGRESSION TASK")
    print("="*60)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_regression(X, y)
    regression_results = train_regressors(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)