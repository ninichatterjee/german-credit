"""
Model evaluation and visualization for German Credit dataset.
Creates plots for EDA and model performance analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression, Ridge

from data import load_raw_data, split_data_classification, split_data_regression
from features import preprocess_for_linear_models


def plot_correlation_heatmap(X, y, headers, save_path='reports/plots_midpointsub/correlation_heatmap.png'):
    """
    Plot correlation heatmap for key numeric features.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target variable
        headers (list): Feature names
        save_path (str): Path to save the plot
    """
    # Create DataFrame with feature names 
    feature_headers = [h for h in headers if h != 'class']
    df = pd.DataFrame(X, columns=feature_headers)
    df['Target'] = y
    
    # All features in this dataset are numeric after encoding
    # Select key numeric features based on domain knowledge
    # Attributes: 2 (Duration), 5 (Credit amount), 8 (Installment rate), 
    #             11 (Present residence), 13 (Age), 16 (Number of credits), 18 (Number of dependents)
    numeric_feature_indices = [1, 4, 7, 10, 12, 15, 17]  # 0-indexed
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
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target variable
        headers (list): Feature names
        save_path (str): Path to save the plot
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
        
        # Prepare data for boxplot (ensure numeric type)
        data_to_plot = [
            df[df['Target'] == 1][feature].astype(float).values,
            df[df['Target'] == 2][feature].astype(float).values
        ]
        
        # Create boxplot
        bp = ax.boxplot(
            data_to_plot,
            tick_labels=['Good Credit (1)', 'Bad Credit (2)'],
            patch_artist=True,
            notch=True,
            showmeans=True
        )
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(feature, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=15)
    
    # Hide unused subplots
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
    
    Args:
        save_path (str): Path to save the plot
    """
    
    # Load and split data
    X, y, headers = load_raw_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_classification(X, y)
    
    # Preprocess
    X_train_prep, X_val_prep, X_test_prep, prep = preprocess_for_linear_models(
        X_train, X_val, X_test
    )
    
    # Train best model (Logistic Regression with optimized parameters)
    model = LogisticRegression(
        C=1.0,
        penalty='l1',
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_prep, y_train)
    
    # Predict on test set
    y_test_pred = model.predict(X_test_prep)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create confusion matrix display
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
    
    # Customize plot
    ax.set_title('Confusion Matrix - Logistic Regression (Test Set)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    
    # Add performance metrics as text
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, pos_label=2)
    recall = recall_score(y_test, y_test_pred, pos_label=2)
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    metrics_text = f'Test Set Performance:\\n'
    metrics_text += f'Accuracy:  {accuracy:.4f}\\n'
    metrics_text += f'Precision: {precision:.4f}\\n'
    metrics_text += f'Recall:    {recall:.4f}\\n'
    metrics_text += f'F1-Score:  {f1:.4f}'
    
    ax.text(1.15, 0.5, metrics_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
    
    return cm, accuracy


def plot_residuals(save_path='reports/plots_midpointsub/residuals_plot.png'):
    """
    Plot residuals vs predicted and residual histogram for the best regression model (Ridge) on test set.
    
    Args:
        save_path (str): Path to save the plot
    """
    
    # Load and split data
    X, y, headers = load_raw_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_regression(X, y)
    
    # Preprocess
    X_train_prep, X_val_prep, X_test_prep, prep = preprocess_for_linear_models(
        X_train, X_val, X_test
    )
    
    # Train best model (Ridge Regression with optimized parameters)
    model = Ridge(alpha=10.0, random_state=42)
    model.fit(X_train_prep, y_train)
    
    # Predict on test set
    y_test_pred = model.predict(X_test_prep)
    
    # Calculate residuals
    residuals = y_test - y_test_pred
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Residuals vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_test_pred, residuals, alpha=0.6, edgecolors='k', s=80)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual Line')
    ax1.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Residuals', fontsize=12, fontweight='bold')
    ax1.set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add trend line
    z = np.polyfit(y_test_pred, residuals, 1)
    p = np.poly1d(z)
    ax1.plot(y_test_pred, p(y_test_pred), "b--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
    ax1.legend()
    
    # Plot 2: Residual Histogram
    ax2 = axes[1]
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Residual')
    ax2.set_xlabel('Residuals', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Residuals', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # Add normal distribution overlay
    from scipy import stats
    mu, std = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, stats.norm.pdf(x, mu, std) * len(residuals) * (residuals.max() - residuals.min()) / 30, 
                  'r-', linewidth=2, label='Normal Distribution')
    ax2_twin.set_ylabel('Probability Density', fontsize=10)
    ax2_twin.legend(loc='upper right')
    
    # Overall title
    fig.suptitle('Residual Analysis - Ridge Regression (Test Set)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Residuals plot saved to {save_path}")
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2 = r2_score(y_test, y_test_pred)
    
    return residuals, mae, rmse, r2


if __name__ == "__main__":
    # Load data
    X, y, headers = load_raw_data()
    print(f"\nLoaded data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Create correlation heatmap
    correlation_matrix = plot_correlation_heatmap(X, y, headers)
    plot_boxplot_summary(X, y, headers)
    print("\nPlots saved to:")
    print("  - reports/plots_midpointsub/correlation_heatmap.png")
    print("  - reports/plots_midpointsub/boxplot_summary.png")
    
    cm, accuracy = plot_confusion_matrix()
    print("\nPlot saved to:")
    print("  - reports/plots_midpointsub/confusion_matrix.png")
    
    residuals, mae, rmse, r2 = plot_residuals()
    print("\nPlot saved to:")
    print("  - reports/plots_midpointsub/residuals_plot.png")