"""
Model evaluation and visualization for German Credit dataset.
Creates plots for EDA and model performance analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, 
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from data import load_raw_data, split_data_classification, split_data_regression
from features import preprocess_for_linear_models, preprocess_for_decision_trees
from config import PLOTS_FINAL_DIR, RANDOM_SEED
from utils import check_for_data_leakage, set_random_seeds

set_random_seeds(RANDOM_SEED)
import pickle
import os


def plot_correlation_heatmap(X, y, headers, save_path='reports/plots_midpointsub/correlation_heatmap.png'):
    """
    Plot correlation heatmap for key numeric features.
    """
    feature_headers = [h for h in headers if h != 'class']
    df = pd.DataFrame(X, columns=feature_headers)
    df['Target'] = y
    
    numeric_feature_indices = [1, 4, 7, 10, 12, 15, 17]
    numeric_feature_names = [headers[i] for i in numeric_feature_indices]
    numeric_df = df[numeric_feature_names + ['Target']]
    
    correlation_matrix = numeric_df.corr()
    
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1
    )
    
    plt.title('Correlation Heatmap - Key Numeric Features', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Features', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Correlation heatmap saved to {save_path}")
    
    return correlation_matrix


def plot_boxplot_summary(X, y, headers, save_path='reports/plots_midpointsub/boxplot_summary.png'):
    """
    Plot boxplot summary for key numeric features, grouped by target.
    """
    feature_headers = [h for h in headers if h != 'class']
    df = pd.DataFrame(X, columns=feature_headers)
    df['Target'] = y
    
    numeric_feature_indices = [1, 4, 7, 10, 12, 15, 17] 
    numeric_feature_names = [headers[i] for i in numeric_feature_indices]
    
    n_features = len(numeric_feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()
    
    for idx, feature in enumerate(numeric_feature_names):
        ax = axes[idx]
        
        data_to_plot = [
            df[df['Target'] == 1][feature].astype(float).values,
            df[df['Target'] == 2][feature].astype(float).values
        ]
        
        bp = ax.boxplot(
            data_to_plot,
            tick_labels=['Good Credit (1)', 'Bad Credit (2)'],
            patch_artist=True,
            notch=True,
            showmeans=True
        )
        
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(feature, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=15)
    
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Boxplot Summary - Key Numeric Features by Target', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Boxplot summary saved to {save_path}")


def plot_confusion_matrix(save_path='reports/plots_midpointsub/confusion_matrix.png'):
    """
    Plot confusion matrix for the best classification model (Logistic Regression) on test set.
    """
    X, y, headers = load_raw_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_classification(X, y)
    
    X_train_prep, X_val_prep, X_test_prep, prep = preprocess_for_linear_models(
        X_train, X_val, X_test
    )
    
    model = LogisticRegression(
        C=1.0,
        penalty='l1',
        solver='liblinear',
        max_iter=1000,
        class_weight='balanced',  # handle class imbalance
        random_state=42
    )
    model.fit(X_train_prep, y_train)
    
    y_test_pred = model.predict(X_test_prep)
    
    cm = confusion_matrix(y_test, y_test_pred)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, pos_label=2)
    recall = recall_score(y_test, y_test_pred, pos_label=2)
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Good Credit (1)', 'Bad Credit (2)']
    )

    disp.plot(
        ax=ax,
        cmap='Blues',
        values_format='d',
        colorbar=True
    )
    
    ax.set_title('Confusion Matrix - Logistic Regression (Test Set)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold', labelpad=10)
    
    metrics_text = 'Test Set Performance:\n\n'
    metrics_text += f'Accuracy:  {accuracy:.4f}\n'
    metrics_text += f'Precision: {precision:.4f}\n'
    metrics_text += f'Recall:    {recall:.4f}\n'
    metrics_text += f'F1-Score:  {f1:.4f}'
    
    ax.text(1.35, 0.5, metrics_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='center',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', 
                     edgecolor='black', alpha=0.8, linewidth=2))
    
    plt.subplots_adjust(right=0.75)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
    
    return cm, accuracy


def plot_residuals(save_path='reports/plots_midpointsub/residuals_plot.png'):
    """
    Plot residuals vs predicted and residual histogram for the best regression model (Linear Regression) on test set.
    """
    
    X, y, headers = load_raw_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_regression(X, y)
    
    X_train_prep, X_val_prep, X_test_prep, prep = preprocess_for_linear_models(
        X_train, X_val, X_test
    )
    
    model = LinearRegression()
    model.fit(X_train_prep, y_train)
    
    y_test_pred = model.predict(X_test_prep)
    
    residuals = y_test - y_test_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1 = axes[0]
    ax1.scatter(y_test_pred, residuals, alpha=0.6, edgecolors='k', s=80)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual Line')
    ax1.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Residuals', fontsize=12, fontweight='bold')
    ax1.set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    z = np.polyfit(y_test_pred, residuals, 1)
    p = np.poly1d(z)
    ax1.plot(y_test_pred, p(y_test_pred), "b--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
    ax1.legend()
    
    ax2 = axes[1]
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Residual')
    ax2.set_xlabel('Residuals', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Residuals', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    from scipy import stats
    mu, std = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, stats.norm.pdf(x, mu, std) * len(residuals) * (residuals.max() - residuals.min()) / 30, 
                  'r-', linewidth=2, label='Normal Distribution')
    ax2_twin.set_ylabel('Probability Density', fontsize=10)
    ax2_twin.legend(loc='upper right')
    
    fig.suptitle('Residual Analysis - Linear Regression (Test Set)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Residuals plot saved to {save_path}")
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    return residuals, mae, rmse


def plot_nn_learning_curves(save_path_clf=f'{PLOTS_FINAL_DIR}/plot1_classification_nn_learning_curve.png',
                            save_path_reg=f'{PLOTS_FINAL_DIR}/plot2_regression_nn_learning_curve.png'):
    """
    Plot 1 & 2: Learning curves for classification and regression neural networks.
    Shows training loss over epochs to monitor overfitting.
    """
    # Load saved loss curves
    with open(f'{PLOTS_FINAL_DIR}/classification_nn_loss_curve.pkl', 'rb') as f:
        clf_loss = pickle.load(f)
    
    with open(f'{PLOTS_FINAL_DIR}/regression_nn_loss_curve.pkl', 'rb') as f:
        reg_loss = pickle.load(f)
    
    # Handle both TensorFlow history (dict) and sklearn loss_curve_ (list)
    if isinstance(clf_loss, dict):
        clf_train_loss = clf_loss['loss']
        clf_val_loss = clf_loss.get('val_loss', None)
    else:
        clf_train_loss = clf_loss
        clf_val_loss = None
    
    # Plot 1: Classification NN
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(clf_train_loss) + 1)
    ax.plot(epochs, clf_train_loss, 'b-', linewidth=2, label='Training Loss')
    if clf_val_loss is not None:
        ax.plot(epochs, clf_val_loss, 'r--', linewidth=2, label='Validation Loss')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Classification Neural Network - Learning Curve', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path_clf, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Classification NN learning curve saved to {save_path_clf}")
    
    # Handle regression loss (could be dict or list)
    if isinstance(reg_loss, dict):
        reg_train_loss = reg_loss['loss']
        reg_val_loss = reg_loss.get('val_loss', None)
    else:
        reg_train_loss = reg_loss
        reg_val_loss = None
    
    # Plot 2: Regression NN
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(reg_train_loss) + 1)
    ax.plot(epochs, reg_train_loss, 'r-', linewidth=2, label='Training Loss')
    if reg_val_loss is not None:
        ax.plot(epochs, reg_val_loss, 'b--', linewidth=2, label='Validation Loss')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    ax.set_title('Regression Neural Network - Learning Curve', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path_reg, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Regression NN learning curve saved to {save_path_reg}")


def plot_best_classification_confusion_matrix(save_path=f'{PLOTS_FINAL_DIR}/plot3_confusion_matrix_best_model.png'):
    """
    Plot 3: Confusion matrix for best classification model on test set.
    Loads the trained model from MLflow (which uses SMOTE).
    """
    from sklearn.metrics import accuracy_score, f1_score
    import mlflow
    
    X, y, headers = load_raw_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_classification(X, y)
    X_train_prep, X_val_prep, X_test_prep, prep = preprocess_for_linear_models(X_train, X_val, X_test)
    
    # Load the Neural Network model from MLflow (trained with SMOTE)
    try:
        # Get the latest run from the NN experiment
        mlflow.set_experiment("german-credit-neural-networks")
        runs = mlflow.search_runs(filter_string="tags.mlflow.runName = 'MLP_Classification'", 
                                   order_by=["start_time DESC"], max_results=1)
        
        if len(runs) > 0:
            run_id = runs.iloc[0]['run_id']
            model_uri = f"runs:/{run_id}/model"
            nn_model = mlflow.sklearn.load_model(model_uri)
            nn_pred = nn_model.predict(X_test_prep)
            nn_f1 = f1_score(y_test, nn_pred, average='weighted')
            best_model_name = 'Neural Network'
            best_pred = nn_pred
            best_f1 = nn_f1
            print(f"Loaded NN model from MLflow run: {run_id}")
        else:
            raise Exception("No MLflow run found")
    except Exception as e:
        # Fallback: train without SMOTE if model not found
        print(f"Warning: Could not load model from MLflow ({e}). Training new model without SMOTE.")
        from sklearn.neural_network import MLPClassifier
        from config import NN_HIDDEN_LAYERS, NN_ACTIVATION, NN_SOLVER, NN_LEARNING_RATE, NN_BATCH_SIZE, NN_ALPHA, NN_MAX_ITER, NN_EARLY_STOPPING, NN_VALIDATION_FRACTION, NN_PATIENCE, RANDOM_SEED
        nn_model = MLPClassifier(hidden_layer_sizes=NN_HIDDEN_LAYERS, activation=NN_ACTIVATION, solver=NN_SOLVER,
                                alpha=NN_ALPHA, batch_size=NN_BATCH_SIZE, learning_rate_init=NN_LEARNING_RATE,
                                max_iter=NN_MAX_ITER, random_state=RANDOM_SEED, early_stopping=NN_EARLY_STOPPING,
                                validation_fraction=NN_VALIDATION_FRACTION, n_iter_no_change=NN_PATIENCE, 
                                shuffle=True, verbose=False)
        nn_model.fit(X_train_prep, y_train)
        nn_pred = nn_model.predict(X_test_prep)
        nn_f1 = f1_score(y_test, nn_pred, average='weighted')
        best_model_name = 'Neural Network (No SMOTE)'
        best_pred = nn_pred
        best_f1 = nn_f1
    
    # Select best model
    lr_model = LogisticRegression(C=1.0, penalty='l1', solver='liblinear', max_iter=1000, 
                                   class_weight='balanced', random_state=42)
    lr_model.fit(X_train_prep, y_train)
    lr_pred = lr_model.predict(X_test_prep)
    lr_f1 = f1_score(y_test, lr_pred, average='weighted')
    
    if nn_f1 >= lr_f1:
        best_model_name = 'Neural Network'
        best_pred = nn_pred
        best_f1 = nn_f1
    else:
        best_model_name = 'Logistic Regression'
        best_pred = lr_pred
        best_f1 = lr_f1
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, best_pred)
    accuracy = accuracy_score(y_test, best_pred)
    precision = precision_score(y_test, best_pred, pos_label=2, zero_division=0)
    recall = recall_score(y_test, best_pred, pos_label=2, zero_division=0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Good Credit (1)', 'Bad Credit (2)'])
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
    
    ax.set_title(f'Confusion Matrix - {best_model_name} (Test Set)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    
    metrics_text = f'Best Model: {best_model_name}\n\n'
    metrics_text += f'Accuracy:  {accuracy:.4f}\n'
    metrics_text += f'Precision: {precision:.4f}\n'
    metrics_text += f'Recall:    {recall:.4f}\n'
    metrics_text += f'F1-Score:  {best_f1:.4f}'
    
    ax.text(1.25, 0.5, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Best model confusion matrix saved to {save_path}")
    print(f"Best classification model: {best_model_name} (F1: {best_f1:.4f})")


def plot_best_regression_residuals(save_path=f'{PLOTS_FINAL_DIR}/plot4_residuals_best_model.png'):
    """
    Plot 4: Residuals vs predicted for best regression model on test set.
    Loads the best model from MLflow instead of retraining.
    """
    import mlflow
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    # Load data
    X, y, headers = load_raw_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_regression(X, y)
    X_train_prep, X_val_prep, X_test_prep, prep = preprocess_for_linear_models(X_train, X_val, X_test)

    # Load best model from MLflow
    mlflow.set_tracking_uri('file:./mlruns')  # Ensure we're looking in the right place
    mlflow.set_experiment('german-credit-regression')
    
    try:
        # Try to find the best run by test_mae
        best_run = mlflow.search_runs(
            order_by=['metrics.test_mae ASC'],
            max_results=1
        ).iloc[0]
        
        # Load the model
        model_uri = f"runs:/{best_run.run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Get predictions
        best_pred = model.predict(X_test_prep)
        best_mae = mean_absolute_error(y_test, best_pred)
        best_model_name = best_run.tags.get('mlflow.runName', 'Best Regression Model')
        
    except Exception as e:
        print(f"Error loading model from MLflow: {e}")
        print("Falling back to training a new Linear Regression model...")
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train_prep, y_train)
        best_pred = model.predict(X_test_prep)
        best_mae = mean_absolute_error(y_test, best_pred)
        best_model_name = 'Linear Regression (Fallback)'
    
    # Calculate residuals
    residuals = y_test - best_pred
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(best_pred, residuals, alpha=0.6, edgecolors='k', s=60)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual')
    ax.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
    ax.set_title(f'Residuals vs Predicted - {best_model_name} (Test Set)', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(best_pred, residuals, 1)
    p = np.poly1d(z)
    ax.plot(best_pred, p(best_pred), 'b--', alpha=0.8, linewidth=2)
    
    # Add metrics text
    metrics_text = f'Best Model: {best_model_name}\n'
    metrics_text += f'MAE: {best_mae:.4f}\n'
    metrics_text += f'RMSE: {rmse:.4f}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Best model residuals plot saved to {save_path}")
    print(f"Best regression model: {best_model_name} (MAE: {best_mae:.4f})")


def plot_feature_importance(save_path=f'{PLOTS_FINAL_DIR}/plot5_feature_importance.png'):
    """
    Plot 5: Feature importance using permutation importance on best classical model.
    Loads the best model from MLflow instead of retraining.
    """
    from sklearn.inspection import permutation_importance
    import mlflow
    import numpy as np

    # Load data
    X, y, headers = load_raw_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_classification(X, y)
    X_train_prep, X_val_prep, X_test_prep, prep = preprocess_for_linear_models(X_train, X_val, X_test)

    # Load best model from MLflow
    mlflow.set_tracking_uri('file:./mlruns')  # Ensure we're looking in the right place
    mlflow.set_experiment('german-credit-classification')
    
    try:
        # Try to find the best run by test_f1_score
        best_run = mlflow.search_runs(
            order_by=['metrics.test_f1_score DESC'],
            max_results=1
        ).iloc[0]
        
        # Load the model
        model_uri = f"runs:/{best_run.run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Get model name for the plot title
        model_name = best_run.tags.get('mlflow.runName', 'Best Model')
        
    except Exception as e:
        print(f"Error loading model from MLflow: {e}")
        print("Falling back to training a new Logistic Regression model...")
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=1.0, penalty='l1', solver='liblinear', max_iter=1000,
                                 class_weight='balanced', random_state=42)
        model.fit(X_train_prep, y_train)
        model_name = 'Logistic Regression (Fallback)'
    
    # Calculate permutation importance on test set
    try:
        perm_importance = permutation_importance(model, X_test_prep, y_test, n_repeats=10, random_state=42)
    except Exception as e:
        print(f"Error calculating permutation importance: {e}")
        print("Falling back to using model coefficients if available...")
        if hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients as importance
            importance = np.abs(model.coef_[0])
            perm_importance = type('obj', (object,), {'importances_mean': importance})
        else:
            # Fallback to random importance if no coefficients
            print("No feature importance available for this model type.")
            return
    
    # Get feature names (after one-hot encoding)
    feature_names = []
    numeric_indices = [1, 4, 7, 10, 12, 15, 17]
    for idx in numeric_indices:
        feature_names.append(headers[idx])
    
    # Add one-hot encoded categorical features
    categorical_indices = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
    for idx in categorical_indices:
        n_categories = len(np.unique(X[:, idx])) - 1  # -1 because of drop='first'
        for i in range(n_categories):
            feature_names.append(f"{headers[idx]}_{i+1}")
    
    # Sort by importance
    importance_means = perm_importance.importances_mean
    indices = np.argsort(importance_means)[::-1][:15]  # Top 15 features
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in indices]
    top_importances = importance_means[indices]
    
    colors = ['green' if x > 0 else 'red' for x in top_importances]
    ax.barh(range(len(indices)), top_importances, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(top_features, fontsize=10)
    ax.set_xlabel('Permutation Importance', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance - Top 15 Features (Logistic Regression)', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {save_path}")


def check_data_integrity():
    """
    Check for data leakage and other integrity issues.
    """
    print("\n" + "="*60)
    print("DATA INTEGRITY CHECK")
    print("="*60)
    
    X, y, headers = load_raw_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_classification(X, y)
    X_train_prep, X_val_prep, X_test_prep, _ = preprocess_for_linear_models(X_train, X_val, X_test)
    
    print("\nChecking for data leakage between train and test sets...")
    check_for_data_leakage(X_train_prep, X_test_prep, threshold=0.95)
    
    print("\nChecking for data leakage between train and validation sets...")
    check_for_data_leakage(X_train_prep, X_val_prep, threshold=0.95)


def generate_final_plots():
    """
    Generate all 5 required plots for the final report.
    """
    print("\n" + "="*60)
    print("GENERATING FINAL REPORT PLOTS")
    print("="*60 + "\n")
    
    # Create directory if it doesn't exist
    os.makedirs(PLOTS_FINAL_DIR, exist_ok=True)
    
    print("Generating Plot 1 & 2: Neural Network Learning Curves...")
    plot_nn_learning_curves()
    
    print("\nGenerating Plot 3: Best Classification Model Confusion Matrix...")
    plot_best_classification_confusion_matrix()
    
    print("\nGenerating Plot 4: Best Regression Model Residuals...")
    plot_best_regression_residuals()
    
    print("\nGenerating Plot 5: Feature Importance...")
    plot_feature_importance()
    
    print("\n" + "="*60)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("="*60)
    print(f"\nPlots saved to: {PLOTS_FINAL_DIR}/")
    print("  - plot1_classification_nn_learning_curve.png")
    print("  - plot2_regression_nn_learning_curve.png")
    print("  - plot3_confusion_matrix_best_model.png")
    print("  - plot4_residuals_best_model.png")
    print("  - plot5_feature_importance.png")


def evaluate_with_cross_validation():
    """
    Evaluate models using 5-fold cross-validation with SMOTE.
    Simple and integrated into the evaluation workflow.
    """
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION EVALUATION")
    print("=" * 60)
    
    # Load data
    X, y, headers = load_raw_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_classification(X, y)
    
    # Combine train + val for CV
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    
    # Preprocess
    X_prep, _, _, _ = preprocess_for_linear_models(X_combined, X_test, X_test)
    
    # Define models
    from config import (
        RANDOM_SEED, NN_HIDDEN_LAYERS, NN_LEARNING_RATE, 
        NN_ALPHA, NN_BATCH_SIZE
    )
    
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, penalty='l1', solver='liblinear',
            class_weight='balanced', random_state=RANDOM_SEED
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=NN_HIDDEN_LAYERS,
            learning_rate_init=NN_LEARNING_RATE,
            alpha=NN_ALPHA,
            batch_size=NN_BATCH_SIZE,
            random_state=RANDOM_SEED,
            max_iter=1000,
            early_stopping=False
        )
    }
    
    # Metrics to evaluate
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # 5-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    print("\nEvaluating models with 5-fold cross-validation...")
    print("(SMOTE applied within each fold)\n")
    
    cv_results = {}
    
    for name, model in models.items():
        print(f"{'='*60}")
        print(f"Model: {name}")
        print(f"{'='*60}")
        
        # Create pipeline with SMOTE
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=RANDOM_SEED)),
            ('classifier', model)
        ])
        
        # Cross-validate
        results = cross_validate(pipeline, X_prep, y_combined, cv=cv, scoring=scoring)
        
        # Store and print results
        cv_results[name] = {}
        for metric in scoring:
            scores = results[f'test_{metric}']
            mean_score = scores.mean()
            std_score = scores.std()
            cv_results[name][metric] = (mean_score, std_score)
            print(f"{metric:12s}: {mean_score:.4f} ± {std_score:.4f}")
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY: Best Model by Metric")
    print("=" * 60)
    for metric in scoring:
        best_model = max(cv_results.items(), key=lambda x: x[1][metric][0])
        print(f"{metric:12s}: {best_model[0]} ({best_model[1][metric][0]:.4f})")
    
    print("\n" + "=" * 60)
    print("Cross-validation complete!")
    print("=" * 60)
    
    return cv_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--final':
        # Generate final report plots
        generate_final_plots()
    elif len(sys.argv) > 1 and sys.argv[1] == '--cv':
        # Run cross-validation
        evaluate_with_cross_validation()
    else:
        # Generate midpoint plots (original behavior)
        X, y, headers = load_raw_data()
        print(f"\nLoaded data: {X.shape[0]} samples, {X.shape[1]} features")
        
        correlation_matrix = plot_correlation_heatmap(X, y, headers)
        plot_boxplot_summary(X, y, headers)
        print("\nPlots saved to:")
        print("  - reports/plots_midpointsub/correlation_heatmap.png")
        print("  - reports/plots_midpointsub/boxplot_summary.png")
        
        cm, accuracy = plot_confusion_matrix()
        print("\nPlot saved to:")
        print("  - reports/plots_midpointsub/confusion_matrix.png")
        
        residuals, mae, rmse = plot_residuals()
        print("\nPlot saved to:")
        print("  - reports/plots_midpointsub/residuals_plot.png")