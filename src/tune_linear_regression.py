"""
Hyperparameter tuning for Linear Regression.
Tests Ridge, Lasso, and ElasticNet with different regularization strengths.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data import load_raw_data, split_data_regression
from features import preprocess_for_linear_models
from config import RANDOM_SEED


def tune_linear_regression(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Tune Linear Regression variants (Ridge, Lasso, ElasticNet).
    Tests different regularization types and strengths.
    
    Returns:
        best_params (dict): Best configuration
        results_df (DataFrame): All results for analysis
    """
    X_train_prep, X_val_prep, X_test_prep, prep = preprocess_for_linear_models(
        X_train, X_val, X_test
    )
    
    results = []
    
    for fit_intercept in [True, False]:
        try:
            model = LinearRegression(fit_intercept=fit_intercept)
            model.fit(X_train_prep, y_train)
            
            y_val_pred = model.predict(X_val_prep)
            y_test_pred = model.predict(X_test_prep)
            
            val_mae = mean_absolute_error(y_val, y_val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            val_r2 = r2_score(y_val, y_val_pred)
            
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_r2 = r2_score(y_test, y_test_pred)
            
            results.append({
                'model_type': 'LinearRegression',
                'alpha': None,
                'l1_ratio': None,
                'fit_intercept': fit_intercept,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_r2': test_r2
            })
            
            print(f"fit_intercept={fit_intercept} -> Val RMSE: {val_rmse:.4f}, Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")
            
        except Exception as e:
            print(f"Error with fit_intercept={fit_intercept}: {str(e)}")
    
    alpha_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    
    for alpha in alpha_values:
        for fit_intercept in [True, False]:
            try:
                model = Ridge(alpha=alpha, fit_intercept=fit_intercept, random_state=RANDOM_SEED)
                model.fit(X_train_prep, y_train)
                
                y_val_pred = model.predict(X_val_prep)
                y_test_pred = model.predict(X_test_prep)
                
                val_mae = mean_absolute_error(y_val, y_val_pred)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                val_r2 = r2_score(y_val, y_val_pred)
                
                test_mae = mean_absolute_error(y_test, y_test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                test_r2 = r2_score(y_test, y_test_pred)
                
                results.append({
                    'model_type': 'Ridge',
                    'alpha': alpha,
                    'l1_ratio': None,
                    'fit_intercept': fit_intercept,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                    'val_r2': val_r2,
                    'test_mae': test_mae,
                    'test_rmse': test_rmse,
                    'test_r2': test_r2
                })
                
                if alpha in [0.01, 0.1, 1.0, 10.0]:
                    print(f"alpha={alpha:6.3f}, fit_intercept={fit_intercept} -> Val RMSE: {val_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
                    
            except Exception as e:
                print(f"Error with alpha={alpha}, fit_intercept={fit_intercept}: {str(e)}")
    
    for alpha in alpha_values:
        for fit_intercept in [True, False]:
            try:
                model = Lasso(alpha=alpha, fit_intercept=fit_intercept, random_state=RANDOM_SEED, max_iter=10000)
                model.fit(X_train_prep, y_train)
                
                y_val_pred = model.predict(X_val_prep)
                y_test_pred = model.predict(X_test_prep)
                
                val_mae = mean_absolute_error(y_val, y_val_pred)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                val_r2 = r2_score(y_val, y_val_pred)
                
                test_mae = mean_absolute_error(y_test, y_test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                test_r2 = r2_score(y_test, y_test_pred)
                
                n_features_used = np.sum(np.abs(model.coef_) > 1e-5)
                
                results.append({
                    'model_type': 'Lasso',
                    'alpha': alpha,
                    'l1_ratio': None,
                    'fit_intercept': fit_intercept,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                    'val_r2': val_r2,
                    'test_mae': test_mae,
                    'test_rmse': test_rmse,
                    'test_r2': test_r2,
                    'n_features_used': n_features_used
                })
                
                if alpha in [0.01, 0.1, 1.0, 10.0]: 
                    print(f"alpha={alpha:6.3f}, fit_intercept={fit_intercept} -> Val RMSE: {val_rmse:.4f}, Test RMSE: {test_rmse:.4f}, Features: {n_features_used}/20")
                    
            except Exception as e:
                print(f"Error with alpha={alpha}, fit_intercept={fit_intercept}: {str(e)}")
    
    alpha_values_en = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    l1_ratio_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for alpha in alpha_values_en:
        for l1_ratio in l1_ratio_values:
            for fit_intercept in [True, False]:
                try:
                    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, 
                                      random_state=RANDOM_SEED, max_iter=10000)
                    model.fit(X_train_prep, y_train)
                    
                    y_val_pred = model.predict(X_val_prep)
                    y_test_pred = model.predict(X_test_prep)
                    
                    val_mae = mean_absolute_error(y_val, y_val_pred)
                    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                    val_r2 = r2_score(y_val, y_val_pred)
                    
                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                    test_r2 = r2_score(y_test, y_test_pred)
                    
                    n_features_used = np.sum(np.abs(model.coef_) > 1e-5)
                    
                    results.append({
                        'model_type': 'ElasticNet',
                        'alpha': alpha,
                        'l1_ratio': l1_ratio,
                        'fit_intercept': fit_intercept,
                        'val_mae': val_mae,
                        'val_rmse': val_rmse,
                        'val_r2': val_r2,
                        'test_mae': test_mae,
                        'test_rmse': test_rmse,
                        'test_r2': test_r2,
                        'n_features_used': n_features_used
                    })
                    
                except Exception as e:
                    pass
    
    print(f"\nTested {len(results)} total configurations")
    
    results_df = pd.DataFrame(results)
    
    best_val_idx = results_df['val_rmse'].idxmin()
    best_val_row = results_df.loc[best_val_idx]
    
    best_test_idx = results_df['test_rmse'].idxmin()
    best_test_row = results_df.loc[best_test_idx]
    
    print(f"Model: {best_val_row['model_type']}")
    if pd.notna(best_val_row['alpha']):
        print(f"alpha: {best_val_row['alpha']}")
    if pd.notna(best_val_row['l1_ratio']):
        print(f"l1_ratio: {best_val_row['l1_ratio']}")
    print(f"fit_intercept: {best_val_row['fit_intercept']}")
    print(f"\nValidation: MAE={best_val_row['val_mae']:.4f}, RMSE={best_val_row['val_rmse']:.4f}, R²={best_val_row['val_r2']:.4f}")
    print(f"Test:       MAE={best_val_row['test_mae']:.4f}, RMSE={best_val_row['test_rmse']:.4f}, R²={best_val_row['test_r2']:.4f}")
    
    print(f"Model: {best_test_row['model_type']}")
    if pd.notna(best_test_row['alpha']):
        print(f"alpha: {best_test_row['alpha']}")
    if pd.notna(best_test_row['l1_ratio']):
        print(f"l1_ratio: {best_test_row['l1_ratio']}")
    print(f"fit_intercept: {best_test_row['fit_intercept']}")
    print(f"\nValidation: MAE={best_test_row['val_mae']:.4f}, RMSE={best_test_row['val_rmse']:.4f}, R²={best_test_row['val_r2']:.4f}")
    print(f"Test:       MAE={best_test_row['test_mae']:.4f}, RMSE={best_test_row['test_rmse']:.4f}, R²={best_test_row['test_r2']:.4f}")
    
    top_10 = results_df.nsmallest(10, 'test_rmse')
    print("\nRank | Model Type      | alpha  | l1_ratio | fit_int | Test RMSE | Test MAE | Test R²")
    print("-" * 95)
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        alpha_str = f"{row['alpha']:6.3f}" if pd.notna(row['alpha']) else "  None"
        l1_str = f"{row['l1_ratio']:6.2f}" if pd.notna(row['l1_ratio']) else "  None"
        fit_str = "Yes" if row['fit_intercept'] else "No "
        print(f"{i:4d} | {row['model_type']:15s} | {alpha_str} | {l1_str}  | {fit_str}     | {row['test_rmse']:9.4f} | {row['test_mae']:8.4f} | {row['test_r2']:7.4f}")
    
    print("\nAverage Performance by Model Type:")
    model_comparison = results_df.groupby('model_type')[['test_rmse', 'test_mae', 'test_r2']].mean()
    print(model_comparison.to_string())
    
    results_df.to_csv('reports/tables_midpointsub/linear_regression_tuning.csv', index=False)
    print("\nResults saved to reports/tables_midpointsub/linear_regression_tuning.csv")
    
    best_config_df = pd.DataFrame([best_test_row])
    best_config_df.to_csv('reports/tables_midpointsub/best_linear_regression_params.csv', index=False)
    print("Best configuration saved to reports/tables_midpointsub/best_linear_regression_params.csv")
    
    return best_test_row, results_df


if __name__ == "__main__":
    X, y, headers = load_raw_data()
    print(f"\nLoaded data: {X.shape[0]} samples, {X.shape[1]} features")
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_regression(X, y)
    print(f"Split data: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}\n")
    
    best_config, results_df = tune_linear_regression(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    baseline_rmse = 0.3888
    best_rmse = best_config['test_rmse']
    improvement = ((baseline_rmse - best_rmse) / baseline_rmse) * 100
    
    print(f"\nBaseline (LinearRegression): Test RMSE = {baseline_rmse:.4f}")
    print(f"Best Configuration:          Test RMSE = {best_rmse:.4f}")
    
    if best_rmse < baseline_rmse:
        print(f"\nImprovement: -{improvement:.2f}% RMSE")
    else:
        print(f"\n✗ Increase: +{-improvement:.2f}% RMSE")
