"""
Hyperparameter tuning for Logistic Regression.
Tunes: C (regularization strength) with L2 penalty
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data import load_raw_data, split_data_classification
from features import preprocess_for_linear_models

RANDOM_SEED = 42


def tune_logistic_regression(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Tune Logistic Regression hyperparameters.
    Tests different C values with L2 penalty.
    
    Args:
        X_train, X_val, X_test: Feature arrays (raw, not preprocessed)
        y_train, y_val, y_test: Target arrays
    
    Returns:
        best_params (dict): Best hyperparameter combination
        results_df (DataFrame): All results for analysis
        best_model: Trained model with best parameters
    """
    X_train_prep, X_val_prep, X_test_prep, prep = preprocess_for_linear_models(
        X_train, X_val, X_test
    )
    
    # smaller C = stronger regularization
    # larger C = weaker regularization
    C_values = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    
    results = []
    
    print(f"C controls regularization strength (smaller = stronger regularization)")
    print()
    
    for C in C_values:
        model = LogisticRegression(
            C=C,
            penalty='l2',
            max_iter=1000,
            random_state=RANDOM_SEED
        )
        
        model.fit(X_train_prep, y_train)
        
        y_val_pred = model.predict(X_val_prep)
        y_val_proba = model.predict_proba(X_val_prep)
        
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        val_roc_auc = roc_auc_score(y_val, y_val_proba[:, 1])
        
        y_test_pred = model.predict(X_test_prep)
        y_test_proba = model.predict_proba(X_test_prep)
        
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        test_roc_auc = roc_auc_score(y_test, y_test_proba[:, 1])
        
        results.append({
            'C': C,
            'penalty': 'l2',
            'val_accuracy': val_accuracy,
            'val_f1_score': val_f1,
            'val_roc_auc': val_roc_auc,
            'test_accuracy': test_accuracy,
            'test_f1_score': test_f1,
            'test_roc_auc': test_roc_auc
        })
        
        print(f"C={C:7.3f} -> Val Acc: {val_accuracy:.4f}, Test Acc: {test_accuracy:.4f}, F1: {test_f1:.4f}, ROC-AUC: {test_roc_auc:.4f}")
    
    results_df = pd.DataFrame(results)
    
    best_idx = results_df['val_accuracy'].idxmax()
    best_C = results_df.loc[best_idx, 'C']
    
    best_params = {
        'C': best_C,
        'penalty': 'l2'
    }
    
    best_val_accuracy = results_df.loc[best_idx, 'val_accuracy']
    best_val_f1 = results_df.loc[best_idx, 'val_f1_score']
    best_val_roc_auc = results_df.loc[best_idx, 'val_roc_auc']
    
    print(f"\nBest Hyperparameters (based on validation accuracy):")
    print(f"  C (regularization): {best_params['C']}")
    print(f"  penalty:            {best_params['penalty']}")
    print(f"\nValidation Performance:")
    print(f"  Accuracy: {best_val_accuracy:.4f}")
    print(f"  F1-Score: {best_val_f1:.4f}")
    print(f"  ROC-AUC:  {best_val_roc_auc:.4f}")

    best_model = LogisticRegression(
        C=best_params['C'],
        penalty=best_params['penalty'],
        max_iter=1000,
        random_state=RANDOM_SEED
    )
    best_model.fit(X_train_prep, y_train)
    
    y_test_pred = best_model.predict(X_test_prep)
    y_test_proba = best_model.predict_proba(X_test_prep)
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_roc_auc = roc_auc_score(y_test, y_test_proba[:, 1])
    
    print(f"\nTest Performance:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    print(f"  ROC-AUC:  {test_roc_auc:.4f}")
    
    results_df.to_csv('reports/tables_midpointsub/logistic_regression_tuning.csv', index=False)
    print(f"\nDetailed results saved to reports/tables_midpointsub/logistic_regression_tuning.csv")
    
    best_params_df = pd.DataFrame([{
        'C': best_params['C'],
        'penalty': best_params['penalty'],
        'val_accuracy': best_val_accuracy,
        'val_f1_score': best_val_f1,
        'val_roc_auc': best_val_roc_auc,
        'test_accuracy': test_accuracy,
        'test_f1_score': test_f1,
        'test_roc_auc': test_roc_auc
    }])
    best_params_df.to_csv('reports/tables_midpointsub/best_logistic_regression_params.csv', index=False)
    print(f"Best parameters saved to reports/tables_midpointsub/best_logistic_regression_params.csv")
    
    return best_params, results_df, best_model


def analyze_C_impact(results_df):
    """
    Analyze the impact of C parameter on performance.
    """
    print("\nC Value  | Val Accuracy | Test Accuracy | Test F1 | Test ROC-AUC")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(f"{row['C']:8.3f} | {row['val_accuracy']:12.4f} | {row['test_accuracy']:13.4f} | {row['test_f1_score']:7.4f} | {row['test_roc_auc']:12.4f}")
    
    best_val_acc = results_df['val_accuracy'].max()
    optimal_C_values = results_df[results_df['val_accuracy'] == best_val_acc]['C'].tolist()
    
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Achieved with C = {optimal_C_values}")
    
    weak_reg = results_df[results_df['C'] >= 10.0]['test_accuracy'].mean()
    strong_reg = results_df[results_df['C'] <= 0.1]['test_accuracy'].mean()
    moderate_reg = results_df[(results_df['C'] > 0.1) & (results_df['C'] < 10.0)]['test_accuracy'].mean()
    
    print(f"Strong regularization (C ≤ 0.1):   Avg test accuracy = {strong_reg:.4f}")
    print(f"Moderate regularization (0.1 < C < 10): Avg test accuracy = {moderate_reg:.4f}")
    print(f"Weak regularization (C ≥ 10):      Avg test accuracy = {weak_reg:.4f}")


if __name__ == "__main__":
    X, y, headers = load_raw_data()
    print(f"\nLoaded data: {X.shape[0]} samples, {X.shape[1]} features")
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_classification(X, y)
    print(f"Split data: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}\n")
    
    best_params, results_df, best_model = tune_logistic_regression(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
analyze_C_impact(results_df)
    
    print("\nBaseline (C=1.0, default): Test Accuracy = 0.7467")
    print(f"Optimized (C={best_params['C']}): Test Accuracy = {results_df.loc[results_df['C'] == best_params['C'], 'test_accuracy'].values[0]:.4f}")
    
    improvement = results_df.loc[results_df['C'] == best_params['C'], 'test_accuracy'].values[0] - 0.7467
    if improvement > 0:
        print(f"\nImprovement: +{improvement*100:.2f}%")
    elif improvement == 0:
        print(f"\n= No change (already optimal)")
    else:
        print(f"\n✗ Decrease: {improvement*100:.2f}%")