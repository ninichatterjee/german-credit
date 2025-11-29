"""
Hyperparameter tuning for Decision Tree Classifier.
Tunes: max_depth, min_samples_split, min_samples_leaf
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data import load_raw_data, split_data_classification
from features import preprocess_for_decision_trees
from config import RANDOM_SEED


def tune_decision_tree_classifier(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Tune Decision Tree Classifier hyperparameters.
    Tests different combinations and finds the best.
    
    Returns:
        best_params (dict): Best hyperparameter combination
        results_df (DataFrame): All results for analysis
    """
    X_train_prep, X_val_prep, X_test_prep, prep = preprocess_for_decision_trees(
        X_train, X_val, X_test
    )
    
    max_depth_values = [3, 5, 7, 10, 15, 20]
    min_samples_split_values = [2, 10, 50, 100]
    min_samples_leaf_values = [1, 5, 20, 50]
    class_weight_values = [None, 'balanced']
    splitter_values = ['best', 'random']
    max_features_values = [None, 'sqrt', 'log2', 0.5]
    
    results = []
    
    print(f"Testing {len(min_samples_split_values)} min_samples_split values...")
    print(f"Testing {len(min_samples_leaf_values)} min_samples_leaf values...")
    print(f"Testing {len(class_weight_values)} class_weight values...")
    print(f"Testing {len(splitter_values)} splitter values...")
    print(f"Testing {len(max_features_values)} max_features values...")
    total_combinations = (len(max_depth_values) * len(min_samples_split_values) * 
                         len(min_samples_leaf_values) * len(class_weight_values) * 
                         len(splitter_values) * len(max_features_values))
    print(f"Total combinations: {total_combinations}")
    
    current = 0
    
    for max_depth in max_depth_values:
        for min_samples_split in min_samples_split_values:
            for min_samples_leaf in min_samples_leaf_values:
                for class_weight in class_weight_values:
                    for splitter in splitter_values:
                        for max_features in max_features_values:
                            current += 1
                            
                            model = DecisionTreeClassifier(
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                class_weight=class_weight,
                                splitter=splitter,
                                max_features=max_features,
                                random_state=RANDOM_SEED
                            )
                            
                            model.fit(X_train_prep, y_train)
                            
                            y_val_pred = model.predict(X_val_prep)
                            y_val_proba = model.predict_proba(X_val_prep)
                            
                            val_accuracy = accuracy_score(y_val, y_val_pred)
                            val_f1 = f1_score(y_val, y_val_pred, average='weighted')
                            val_roc_auc = roc_auc_score(y_val, y_val_proba[:, 1])
                            
                            results.append({
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split,
                                'min_samples_leaf': min_samples_leaf,
                                'class_weight': class_weight,
                                'splitter': splitter,
                                'max_features': max_features,
                                'val_accuracy': val_accuracy,
                                'val_f1_score': val_f1,
                                'val_roc_auc': val_roc_auc
                            })
                            
                            if current % 200 == 0:
                                print(f"Progress: {current}/{total_combinations} combinations tested...")
    
    results_df = pd.DataFrame(results)
    
    best_idx = results_df['val_accuracy'].idxmax()
    max_depth_val = results_df.loc[best_idx, 'max_depth']
    best_params = {
        'max_depth': int(max_depth_val) if pd.notna(max_depth_val) else None,
        'min_samples_split': int(results_df.loc[best_idx, 'min_samples_split']),
        'min_samples_leaf': int(results_df.loc[best_idx, 'min_samples_leaf']),
        'class_weight': results_df.loc[best_idx, 'class_weight'],
        'splitter': results_df.loc[best_idx, 'splitter'],
        'max_features': results_df.loc[best_idx, 'max_features']
    }
    best_val_accuracy = results_df.loc[best_idx, 'val_accuracy']
    best_val_f1 = results_df.loc[best_idx, 'val_f1_score']
    best_val_roc_auc = results_df.loc[best_idx, 'val_roc_auc']
    
    print(f"\nBest Hyperparameters (based on validation accuracy):")
    print(f"  max_depth:         {best_params['max_depth']}")
    print(f"  min_samples_split: {best_params['min_samples_split']}")
    print(f"  min_samples_leaf:  {best_params['min_samples_leaf']}")
    print(f"  class_weight:      {best_params['class_weight']}")
    print(f"  splitter:          {best_params['splitter']}")
    print(f"  max_features:      {best_params['max_features']}")
    print(f"\nValidation Performance:")
    print(f"  Accuracy: {best_val_accuracy:.4f}")
    print(f"  F1-Score: {best_val_f1:.4f}")
    print(f"  ROC-AUC:  {best_val_roc_auc:.4f}")
    
    best_model = DecisionTreeClassifier(
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        class_weight=best_params['class_weight'],
        splitter=best_params['splitter'],
        max_features=best_params['max_features'],
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
    
    results_df.to_csv('reports/tables_midpointsub/hyperparameter_tuning_results.csv', index=False)
    print(f"\nDetailed results saved to reports/tables_midpointsub/hyperparameter_tuning_results.csv")
    
    best_params_df = pd.DataFrame([{
        'max_depth': best_params['max_depth'],
        'min_samples_split': best_params['min_samples_split'],
        'min_samples_leaf': best_params['min_samples_leaf'],
        'val_accuracy': best_val_accuracy,
        'val_f1_score': best_val_f1,
        'val_roc_auc': best_val_roc_auc,
        'test_accuracy': test_accuracy,
        'test_f1_score': test_f1,
        'test_roc_auc': test_roc_auc
    }])
    best_params_df.to_csv('reports/tables_midpointsub/best_hyperparameters.csv', index=False)
    print(f"Best parameters saved to reports/tables_midpointsub/best_hyperparameters.csv")
    
    return best_params, results_df, best_model


def analyze_hyperparameter_impact(results_df):
    """
    Analyze the impact of each hyperparameter on performance.
    """
    print("\n--- Impact of max_depth ---")
    max_depth_impact = results_df.groupby('max_depth')['val_accuracy'].agg(['mean', 'std', 'max'])
    print(max_depth_impact.to_string())
    
    print("\n--- Impact of min_samples_split ---")
    min_samples_split_impact = results_df.groupby('min_samples_split')['val_accuracy'].agg(['mean', 'std', 'max'])
    print(min_samples_split_impact.to_string())
    
    print("\n--- Impact of min_samples_leaf ---")
    min_samples_leaf_impact = results_df.groupby('min_samples_leaf')['val_accuracy'].agg(['mean', 'std', 'max'])
    print(min_samples_leaf_impact.to_string())
    
    print("\n--- Impact of class_weight ---")
    class_weight_impact = results_df.groupby('class_weight')['val_accuracy'].agg(['mean', 'std', 'max'])
    print(class_weight_impact.to_string())
    
    print("\n--- Impact of splitter ---")
    splitter_impact = results_df.groupby('splitter')['val_accuracy'].agg(['mean', 'std', 'max'])
    print(splitter_impact.to_string())
    
    print("\n--- Impact of max_features ---")
    max_features_impact = results_df.groupby('max_features')['val_accuracy'].agg(['mean', 'std', 'max'])
    print(max_features_impact.to_string())
    
    print("\n--- Top 10 Hyperparameter Combinations ---")
    top_10 = results_df.nlargest(10, 'val_accuracy')[['max_depth', 'min_samples_split', 'min_samples_leaf', 'class_weight', 'splitter', 'max_features', 'val_accuracy', 'val_f1_score', 'val_roc_auc']]
    print(top_10.to_string(index=False))


if __name__ == "__main__":
    X, y, headers = load_raw_data()
    print(f"\nLoaded data: {X.shape[0]} samples, {X.shape[1]} features")
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_classification(X, y)
    print(f"Split data: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}\n")
    
    best_params, results_df, best_model = tune_decision_tree_classifier(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    analyze_hyperparameter_impact(results_df)
