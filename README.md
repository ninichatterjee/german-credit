# German Credit Risk Prediction

A machine learning project for predicting credit risk using the German Credit dataset from the UCI Machine Learning Repository. This project implements baseline models (Logistic Regression, Decision Trees) with MLflow experiment tracking, and will be extended with neural networks for the final submission.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation & Setup](#installation--setup)
- [Reproducibility](#reproducibility)
- [Current Status](#current-status)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Results](#results)

---

## Project Overview

### Objective
Predict credit risk (Good vs Bad credit) using customer financial and demographic data. The project addresses:
- **Classification Task**: Binary classification (Good/Bad credit)
- **Regression Task**: Predicting credit rating on a continuous scale (1.0 to 2.0)

### Key Challenges Addressed
1. **Class Imbalance**: 70% Good credit vs 30% Bad credit
   - Solution: Used `class_weight='balanced'` in Logistic Regression
   - Prioritized recall for bad credits (catching defaults is more costly)

2. **Multicollinearity**: One-hot encoding created redundant features
   - Solution: Added `drop='first'` to OneHotEncoder
   - Reduced features from 61 to 48, achieving full matrix rank

3. **Model Selection**: Linear vs non-linear models
   - Finding: Linear models (Logistic/Linear Regression) outperformed Decision Trees
   - Insight: Credit risk has predominantly linear relationships with features

4. **Hyperparameter Tuning**: Balancing accuracy vs recall
   - Decision: Kept baseline config (C=1.0, penalty=l1, class_weight=balanced)
   - Rationale: 71.3% accuracy with 73.3% recall for bad credits (optimal for credit risk)

### Design Decisions

**Why Logistic Regression as baseline?**
- Linear and interpretable (important for financial decisions)
- Handles class imbalance with `class_weight='balanced'`
- L1 regularization provides automatic feature selection
- Fast training and prediction

**Why these metrics?**
- **Accuracy**: Overall correctness measure
- **F1-Score**: Balances precision and recall (important for imbalanced data)
- **ROC-AUC**: Measures discrimination ability across all thresholds
- **Recall (Bad Credit)**: Most critical - missing a bad credit is costlier than rejecting a good one

**Why exclude Naive Bayes?**
- Dataset has many continuous features requiring Gaussian assumptions
- Assumes feature independence (violated by correlated financial features)
- Logistic Regression provides better baseline for this use case

---

## Dataset

**Source**: [UCI Machine Learning Repository - German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)

**Statistics**:
- **Samples**: 1,000 credit applications
- **Features**: 20 (7 numerical, 13 categorical)
- **Target**: Binary (1 = Good credit, 2 = Bad credit)
- **Class Distribution**: 70% Good, 30% Bad
- **Missing Values**: None
- **Duplicates**: None

**Key Features**:
- Credit amount, duration, installment rate
- Checking account status, savings account
- Employment duration, age, number of dependents
- Housing status, job type, credit history

**Data Location**: `data/` directory (automatically downloaded from UCI)

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/german-credit.git
cd german-credit
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: Machine learning models
- `mlflow`: Experiment tracking
- `matplotlib`, `seaborn`: Visualization
- `ucimlrepo`: UCI dataset loader

### Step 3: Verify Installation
```bash
python3 -c "import mlflow, sklearn, pandas; print('✅ All dependencies installed')"
```

---

## Reproducibility

### Running the Complete Pipeline

#### 1. Train Baseline Models
```bash
python3 src/train_baselines.py
```

**What this does**:
- Loads German Credit data from UCI
- Trains 4 baseline models:
  - Logistic Regression (classification)
  - Decision Tree Classifier
  - Linear Regression (regression)
  - Decision Tree Regressor
- Logs all experiments to MLflow
- Saves results to `reports/tables_midpointsub/`

**Expected Output**:
```
Loaded data: 1000 samples, 20 features

Training Logistic Regression...
  Test Accuracy: 0.7133
Training Decision Tree Classifier...
  Test Accuracy: 0.5467

Training Linear Regression...
  Test RMSE: 0.3879
Training Decision Tree Regressor...
  Test RMSE: 0.4144

Saved: classification_metrics.csv
Saved: regression_metrics.csv
```

#### 2. Hyperparameter Tuning (Optional)
```bash
python3 tune_logistic_regression.py
```

**What this does**:
- Performs GridSearchCV with 5-fold cross-validation
- Tests 144 parameter combinations
- Logs best model to MLflow
- Saves detailed results to `reports/logistic_regression_tuning_results.csv`

#### 3. View Results in MLflow UI
```bash
mlflow ui
```

Then open your browser to: **http://127.0.0.1:5000**

### Random Seed
All models use `RANDOM_SEED = 42` for reproducibility. Results should be identical across runs.

### Data Splits
- **Training**: 70% (700 samples)
- **Validation**: 15% (150 samples)
- **Test**: 15% (150 samples)

Splits are stratified to maintain class balance.

---

## Current Status

### Completed

1. **Data Pipeline**
   - Automated data loading from UCI repository
   - Stratified train/val/test splits
   - Feature engineering (6 engineered features)
   - Preprocessing pipelines for linear and tree-based models

2. **Baseline Models**
   - Logistic Regression: 71.33% accuracy, 73.3% recall (bad credits)
   - Decision Tree Classifier: 54.67% accuracy
   - Linear Regression: RMSE 0.3879
   - Decision Tree Regressor: RMSE 0.4144

3. **Hyperparameter Tuning**
   - GridSearchCV with 144 configurations tested
   - Found optimal parameters (kept baseline for better recall)
   - Documented trade-offs between accuracy and recall

4. **Experiment Tracking**
   - MLflow integration with 13 logged runs
   - All hyperparameters and metrics tracked
   - Models saved for deployment

5. **Analysis & Documentation**
   - Comprehensive model comparison
   - Multicollinearity analysis and fix
   - Business context considerations (cost of false negatives)

### Planned for Final Submission

1. **Neural Network Implementation**
   - Feedforward neural network for classification
   - Hyperparameter tuning (layers, neurons, dropout)
   - Comparison with baseline models

2. **Advanced Models**
   - Random Forest
   - Gradient Boosting (XGBoost/LightGBM)
   - Ensemble methods

3. **Model Deployment**
   - Model serialization
   - Inference pipeline
   - API endpoint (optional)

---

## MLflow Experiment Tracking

### Viewing Experiments

**Start MLflow UI**:
```bash
cd /path/to/german-credit
mlflow ui
```

**Access UI**: Open browser to `http://127.0.0.1:5000`

### Experiments Tracked

### What's Logged

**For each run**:
- ✅ All hyperparameters (C, penalty, solver, max_depth, etc.)
- ✅ Performance metrics (accuracy, F1, ROC-AUC, RMSE, MAE)
- ✅ Validation and test set results
- ✅ Model artifacts (.pkl files)
- ✅ Preprocessing type (one-hot vs label encoding)
- ✅ Model type (classification vs regression)

### Artifact Storage

**Location**: `mlruns/` directory (local filesystem)

### Loading Saved Models

```python
import mlflow

# Load a specific model by run ID
model = mlflow.sklearn.load_model("runs:/<run_id>/model")

# Make predictions
predictions = model.predict(X_test)
```

### Comparing Runs

1. Open MLflow UI
2. Select experiment: `german-credit-baseline-models`
3. Check multiple runs
4. Click "Compare" button
5. View side-by-side metrics and parameters

---

## Results

### Classification Task

| Model | Test Accuracy | F1-Score | ROC-AUC | Recall (Bad) |
|-------|---------------|----------|---------|-------------|
| **Logistic Regression** | **71.33%** | **0.723** | **0.779** | **73.3%** ⭐ |
| Decision Tree | 54.67% | 0.555 | 0.693 | 62.2% |

**Winner**: Logistic Regression
- Best balance between accuracy and recall
- Catches 73.3% of bad credits (critical for credit risk)
- Interpretable coefficients for business understanding

### Regression Task

| Model | Test RMSE | Test MAE | Val RMSE | Generalization |
|-------|-----------|----------|----------|----------------|
| **Linear Regression** | **0.3879** | **0.3313** | 0.4177 | Excellent |
| Decision Tree | 0.4145 | 0.3393 | 0.4347 | Good |

**Winner**: Linear Regression
- 6.4% better RMSE than Decision Tree
- No overfitting (test performs better than validation)
- Simple and interpretable

### Key Insights

1. **Linear models dominate**: Credit risk has strong linear relationships
2. **No overfitting**: All models generalize well (val ≈ test performance)
3. **Class imbalance handled**: `class_weight='balanced'` effective
4. **Moderate task difficulty**: 71% accuracy suggests inherent noise in credit data
5. **Recall prioritized**: For credit risk, catching bad credits > overall accuracy

### Hyperparameter Tuning Results

| Configuration | Accuracy | Recall (Bad) | Verdict |
|---------------|----------|--------------|----------|
| Baseline (l1, balanced) | 71.33% | 73.3% | ✅ Best overall |
| Tuned (l2, no balance) | 77.33% | 51.1% | ❌ Poor recall |
| Hybrid (l2, balanced) | 70.00% | 71.1% | ⚠️ Worse than baseline |

**Decision**: Keep baseline configuration
- Optimal balance for credit risk application
- L1 regularization provides feature selection
- `class_weight='balanced'` essential for catching bad credits

---

### Potential Improvements

- Feature engineering: polynomial features, interactions
- Cost-sensitive learning: custom loss functions
- Threshold optimization: adjust decision boundary
- Cross-validation: k-fold for robust estimates
- Feature selection: remove noisy features

## 📚 References

1. **Dataset**: Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77

2. **MLflow**: https://mlflow.org/docs/latest/index.html

3. **Scikit-learn**: Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830.

**Last Updated**: November 2, 2025