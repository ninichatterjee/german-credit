"""
Hyperparameter tuning for Decision Tree Regressor.
Tunes: max_depth, min_samples_split, min_samples_leaf
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data import load_raw_data, split_data_regression
from features import preprocess_for_decision_trees
from config import RANDOM_SEED


def tune_decision_tree_regressor(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Tune Decision Tree Regressor hyperparameters.
    Tests different combinations and finds the best.
    
    Returns:
        best_params (dict): Best hyperparameter combination
        results_df (DataFrame): All results for analysis
    """
    X_train_prep, X_val_prep, X_test_prep, prep = preprocess_for_decision_trees(
        X_train, X_val, X_test
    )
    
    max_depth_values = list(range(2, 10)) 
    min_samples_split_values = list(range(90, 110))
    min_samples_leaf_values = list(range(1, 12))
    ccp_alpha_values = [round(0.00025 + i * 0.0001, 5) for i in range(6)]
    
    results = []

    print(f"Testing {len(min_samples_split_values)} min_samples_split values (90 to 109)...")
    print(f"Testing {len(min_samples_leaf_values)} min_samples_leaf values (15 to 24)...")
    print(f"Testing {len(ccp_alpha_values)} ccp_alpha values (0.00025 to 0.00075)...")
    total_combinations = (len(max_depth_values) * len(min_samples_split_values) * 
                         len(min_samples_leaf_values) * len(ccp_alpha_values))
    print(f"Total combinations: {total_combinations}")

    current = 0
    
    for max_depth in max_depth_values:
        for min_samples_split in min_samples_split_values:
            for min_samples_leaf in min_samples_leaf_values:
                for ccp_alpha in ccp_alpha_values:
                    current += 1
                    
                    model = DecisionTreeRegressor(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        ccp_alpha=ccp_alpha,
                        random_state=RANDOM_SEED
                    )
                
                    model.fit(X_train_prep, y_train)
                    
                    y_val_pred = model.predict(X_val_prep)
                    
                    val_mae = mean_absolute_error(y_val, y_val_pred)
                    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                    val_r2 = r2_score(y_val, y_val_pred)
                    
                    y_test_pred = model.predict(X_test_prep)
                    
                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                    test_r2 = r2_score(y_test, y_test_pred)
                    
                    results.append({
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'ccp_alpha': ccp_alpha,
                        'val_mae': val_mae,
                        'val_rmse': val_rmse,
                        'val_r2': val_r2,
                        'test_mae': test_mae,
                        'test_rmse': test_rmse,
                        'test_r2': test_r2
                    })
                    
                    if current % 20 == 0:
                        print(f"Progress: {current}/{total_combinations} combinations tested...")
    
    results_df = pd.DataFrame(results)
    
    best_idx = results_df['val_rmse'].idxmin()
    best_params = {
        'max_depth': int(results_df.loc[best_idx, 'max_depth']),
        'min_samples_split': int(results_df.loc[best_idx, 'min_samples_split']),
        'min_samples_leaf': int(results_df.loc[best_idx, 'min_samples_leaf']),
        'ccp_alpha': float(results_df.loc[best_idx, 'ccp_alpha'])
    }
    best_val_mae = results_df.loc[best_idx, 'val_mae']
    best_val_rmse = results_df.loc[best_idx, 'val_rmse']
    best_val_r2 = results_df.loc[best_idx, 'val_r2']

    print(f"\nBest Hyperparameters (based on validation RMSE):")
    print(f"  max_depth:         {best_params['max_depth']}")
    print(f"  min_samples_split: {best_params['min_samples_split']}")
    print(f"  min_samples_leaf:  {best_params['min_samples_leaf']}")
    print(f"  ccp_alpha:         {best_params['ccp_alpha']}")
    print(f"\nValidation Performance:")
    print(f"  MAE:  {best_val_mae:.4f}")
    print(f"  RMSE: {best_val_rmse:.4f}")
    print(f"  R²:   {best_val_r2:.4f}")
    
    best_model = DecisionTreeRegressor(
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        ccp_alpha=best_params['ccp_alpha'],
        random_state=RANDOM_SEED
    )
    best_model.fit(X_train_prep, y_train)
    
    y_test_pred = best_model.predict(X_test_prep)
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTest Performance:")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    
    results_df.to_csv('reports/tables_midpointsub/decision_tree_regressor_tuning.csv', index=False)
    print(f"\nDetailed results saved to reports/tables_midpointsub/decision_tree_regressor_tuning.csv")
    
    best_params_df = pd.DataFrame([{
        'max_depth': best_params['max_depth'],
        'min_samples_split': best_params['min_samples_split'],
        'min_samples_leaf': best_params['min_samples_leaf'],
        'ccp_alpha': best_params['ccp_alpha'],
        'val_mae': best_val_mae,
        'val_rmse': best_val_rmse,
        'val_r2': best_val_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2
    }])
    best_params_df.to_csv('reports/tables_midpointsub/best_decision_tree_regressor_params.csv', index=False)
    print(f"Best parameters saved to reports/tables_midpointsub/best_decision_tree_regressor_params.csv")
    
    return best_params, results_df, best_model


def analyze_hyperparameter_impact(results_df):
    """
    Analyze the impact of each hyperparameter on performance.
    """
    print("\n--- Impact of max_depth ---")
    max_depth_impact = results_df.groupby('max_depth')[['val_rmse', 'test_rmse', 'val_r2']].mean()
    print(max_depth_impact.to_string())
    
    print("\n--- Impact of min_samples_split ---")
    min_samples_split_impact = results_df.groupby('min_samples_split')[['val_rmse', 'test_rmse', 'val_r2']].mean()
    print(min_samples_split_impact.to_string())
    
    print("\n--- Impact of min_samples_leaf ---")
    min_samples_leaf_impact = results_df.groupby('min_samples_leaf')[['val_rmse', 'test_rmse', 'val_r2']].mean()
    print(min_samples_leaf_impact.to_string())
    
    print("\n--- Impact of ccp_alpha ---")
    ccp_alpha_impact = results_df.groupby('ccp_alpha')[['val_rmse', 'test_rmse', 'val_r2']].mean()
    print(ccp_alpha_impact.to_string())
    
    print("\n--- Top 10 Hyperparameter Combinations (by validation RMSE) ---")
    top_10 = results_df.nsmallest(10, 'val_rmse')[['max_depth', 'min_samples_split', 'min_samples_leaf', 'ccp_alpha', 'val_rmse', 'test_rmse', 'val_r2', 'test_r2']]
    print(top_10.to_string(index=False))


if __name__ == "__main__":
    X, y, headers = load_raw_data()
    print(f"\nLoaded data: {X.shape[0]} samples, {X.shape[1]} features")
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_regression(X, y)
    print(f"Split data: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}\n")
    
    best_params, results_df, best_model = tune_decision_tree_regressor(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    analyze_hyperparameter_impact(results_df)
    
    print("\nBaseline (default parameters): Test RMSE = 0.5477, Test MAE = 0.3000")
    print(f"Optimized: Test RMSE = {results_df.loc[results_df['val_rmse'].idxmin(), 'test_rmse']:.4f}, Test MAE = {results_df.loc[results_df['val_rmse'].idxmin(), 'test_mae']:.4f}")
    
    baseline_rmse = 0.5477
    best_rmse = results_df.loc[results_df['val_rmse'].idxmin(), 'test_rmse']
    improvement = ((baseline_rmse - best_rmse) / baseline_rmse) * 100
    
    if best_rmse < baseline_rmse:
        print(f"\nImprovement: -{improvement:.2f}% RMSE (lower is better)")
    else:
        print(f"\n✗ Increase: +{-improvement:.2f}% RMSE")
