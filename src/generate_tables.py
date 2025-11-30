"""
Generate comparison tables for Classical ML vs Neural Networks.
Creates Table 1 (Classification) and Table 2 (Regression) for the final report.
"""
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error
import numpy as np

from data import load_raw_data, split_data_classification, split_data_regression
from features import preprocess_for_linear_models
from config import TABLES_FINAL_DIR


def generate_comparison_tables():
    """Generate Tables 1 and 2 comparing Classical ML vs NN performance."""
    
    print("\n" + "="*60)
    print("GENERATING COMPARISON TABLES")
    print("="*60)
    
    X, y, headers = load_raw_data()
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_classification(X, y)
    X_train_prep, X_val_prep, X_test_prep, _ = preprocess_for_linear_models(X_train, X_val, X_test)
    
    mlflow.set_experiment("german-credit-baseline-models")
    lr_runs = mlflow.search_runs(filter_string="tags.mlflow.runName = 'Logistic_Regression'", 
                                  order_by=["start_time DESC"], max_results=1)
    
    mlflow.set_experiment("german-credit-neural-networks")
    nn_runs = mlflow.search_runs(filter_string="tags.mlflow.runName = 'TF_MLP_Classification'", 
                                  order_by=["start_time DESC"], max_results=1)
    if len(nn_runs) == 0:
        nn_runs = mlflow.search_runs(filter_string="tags.mlflow.runName = 'MLP_Classification'", 
                                      order_by=["start_time DESC"], max_results=1)
    
    classification_data = []
    
    if len(lr_runs) > 0:
        lr_metrics = {
            'Model': 'Logistic Regression',
            'Val_Accuracy': lr_runs.iloc[0]['metrics.val_accuracy'],
            'Val_F1': lr_runs.iloc[0]['metrics.val_f1_score'],
            'Val_ROC_AUC': lr_runs.iloc[0]['metrics.val_roc_auc'],
            'Test_Accuracy': lr_runs.iloc[0]['metrics.test_accuracy'],
            'Test_F1': lr_runs.iloc[0]['metrics.test_f1_score'],
            'Test_ROC_AUC': lr_runs.iloc[0]['metrics.test_roc_auc']
        }
        classification_data.append(lr_metrics)
    
    if len(nn_runs) > 0:
        nn_metrics = {
            'Model': 'Neural Network',
            'Val_Accuracy': nn_runs.iloc[0]['metrics.val_accuracy'],
            'Val_F1': nn_runs.iloc[0]['metrics.val_f1_score'],
            'Val_ROC_AUC': nn_runs.iloc[0]['metrics.val_roc_auc'],
            'Test_Accuracy': nn_runs.iloc[0]['metrics.test_accuracy'],
            'Test_F1': nn_runs.iloc[0]['metrics.test_f1_score'],
            'Test_ROC_AUC': nn_runs.iloc[0]['metrics.test_roc_auc']
        }
        classification_data.append(nn_metrics)
    
    table1 = pd.DataFrame(classification_data)
    table1 = table1.round(4)
    table1.to_csv(f'{TABLES_FINAL_DIR}/table1_classification_comparison.csv', index=False)
    
    print("\nTable 1: Classification Comparison")
    print(table1.to_string(index=False))
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_regression(X, y)
    
    mlflow.set_experiment("german-credit-baseline-models")
    linreg_runs = mlflow.search_runs(filter_string="tags.mlflow.runName = 'Linear_Regression'", 
                                      order_by=["start_time DESC"], max_results=1)
    
    mlflow.set_experiment("german-credit-neural-networks")
    nn_reg_runs = mlflow.search_runs(filter_string="tags.mlflow.runName = 'MLP_Regression'", 
                                      order_by=["start_time DESC"], max_results=1)
    
    regression_data = []
    
    if len(linreg_runs) > 0:
        linreg_metrics = {
            'Model': 'Linear Regression',
            'Val_MAE': linreg_runs.iloc[0]['metrics.val_mae'],
            'Val_RMSE': linreg_runs.iloc[0]['metrics.val_rmse'],
            'Test_MAE': linreg_runs.iloc[0]['metrics.test_mae'],
            'Test_RMSE': linreg_runs.iloc[0]['metrics.test_rmse']
        }
        regression_data.append(linreg_metrics)
    
    if len(nn_reg_runs) > 0:
        nn_reg_metrics = {
            'Model': 'Neural Network',
            'Val_MAE': nn_reg_runs.iloc[0]['metrics.val_mae'],
            'Val_RMSE': nn_reg_runs.iloc[0]['metrics.val_rmse'],
            'Test_MAE': nn_reg_runs.iloc[0]['metrics.test_mae'],
            'Test_RMSE': nn_reg_runs.iloc[0]['metrics.test_rmse']
        }
        regression_data.append(nn_reg_metrics)
    
    table2 = pd.DataFrame(regression_data)
    table2 = table2.round(4)
    table2.to_csv(f'{TABLES_FINAL_DIR}/table2_regression_comparison.csv', index=False)
    
    print("\nTable 2: Regression Comparison")
    print(table2.to_string(index=False))
    
    print(f"\nTables saved to {TABLES_FINAL_DIR}/")
    print("  - table1_classification_comparison.csv")
    print("  - table2_regression_comparison.csv")
    
    return table1, table2


if __name__ == "__main__":
    generate_comparison_tables()
