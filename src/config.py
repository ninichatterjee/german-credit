# All constants and hyperparameters are defined here for centralized consistency.

# Random seed for reproducibility
RANDOM_SEED = 42

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Feature indices
CATEGORICAL_INDICES = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
NUMERICAL_INDICES = [1, 4, 7, 10, 12, 15, 17]

# MLflow experiment names
MLFLOW_EXPERIMENT_BASELINE = "german-credit-baseline-models"
MLFLOW_EXPERIMENT_NN = "german-credit-neural-networks"
MLFLOW_EXPERIMENT_TUNING = "german-credit-hyperparameter-tuning"

# Neural network hyperparameters (optimized from tuning with SMOTE)
NN_HIDDEN_LAYERS = (64, 32, 16)  # Three layers - best architecture
NN_ACTIVATION = 'relu'
NN_SOLVER = 'adam'
NN_LEARNING_RATE = 0.001  # Reduced from 0.007 to prevent numeric instability
NN_BATCH_SIZE = 32  # Increased from 12 for more stable gradients
NN_ALPHA = 0.001  # Increased L2 regularization to prevent overfitting
NN_MAX_ITER = 1000
NN_EARLY_STOPPING = True
NN_VALIDATION_FRACTION = 0.2
NN_PATIENCE = 10

# Paths
REPORTS_DIR = 'reports'
PLOTS_FINAL_DIR = 'reports/plots_final'
PLOTS_MIDPOINT_DIR = 'reports/plots_midpointsub'
TABLES_FINAL_DIR = 'reports/tables_final'
TABLES_MIDPOINT_DIR = 'reports/tables_midpointsub'
