# Data Manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt

# Preprocessing & Data Splitting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Regression Models
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV

# Classification Models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import xgboost as xgb

# Model Evaluation
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.inspection import DecisionBoundaryDisplay

# Model Persistence
import joblib
import os

#///////////////////////////////////////////////////////////////////////////////


# --- Load Data from CSV ---
# Assumes data is in a file named 'dataset.csv'
try:
    df = pd.read_csv('dataset.csv')
    X = df.drop('target_column', axis=1)
    y = df['target_column']
except FileNotFoundError:
    print("Error: 'dataset.csv' not found.")
    # Create dummy data if file not found, for demonstration
    X = pd.DataFrame({
        'numeric_feature_1': np.random.rand(100),
        'numeric_feature_2': np.random.rand(100) * 10,
        'categorical_feature': np.random.choice(['A', 'B', 'C'], 100)
    })
    y = pd.Series(np.random.randint(0, 2, 100))


# --- Comprehensive Preprocessing for Mixed Data Types ---
# Identify numerical and categorical columns automatically
numerical_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# Create a preprocessing pipeline
# - StandardScaler scales numerical features (mean=0, variance=1)
# - OneHotEncoder converts categorical features into numerical format
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Apply the preprocessing steps
X_processed = preprocessor.fit_transform(X)


# --- Split Data into Training and Testing Sets ---
# test_size=0.2 means 20% of the data is for testing
# random_state=42 ensures the split is the same every time
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

print(f"Original data shape: {X.shape}")
print(f"Processed training data shape: {X_train.shape}")
print(f"Processed testing data shape: {X_test.shape}")


#///////////////////////////////////////////////////////////////////////////////


# --- Calculate Basic Descriptive Statistics ---
# Note: Use the original DataFrame `df` before processing
# Example for a specific column 'x1' and 'y1'
stats_summary = {
    'Mean (x)': df['x1'].mean(),
    'Variance (x)': df['x1'].var(),
    'Mean (y)': df['y1'].mean(),
    'Variance (y)': df['y1'].var(),
    'Correlation': df['x1'].corr(df['y1'])
}

# Create a DataFrame for nice printing
stats_df = pd.DataFrame([stats_summary])
print("Descriptive Statistics:")
print(stats_df.round(3))


#///////////////////////////////////////////////////////////////////////////////


# --- Linear Regression (Baseline) ---
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_preds = lin_model.predict(X_test)
lin_rmse = root_mean_squared_error(y_test, lin_preds)
print(f"Linear Regression RMSE: {lin_rmse:.4f}")
# print(f"Coefficients: {lin_model.coef_}")


# --- Lasso (L1) Regularized Regression ---
# Lasso can shrink some feature coefficients to zero, effectively performing feature selection.
# cv=5 means 5-fold cross-validation
lasso_model = LassoCV(cv=5, random_state=42)
lasso_model.fit(X_train, y_train)
lasso_preds = lasso_model.predict(X_test)
lasso_rmse = root_mean_squared_error(y_test, lasso_preds)
print(f"Lasso Regression RMSE: {lasso_rmse:.4f}")
print(f"Lasso Best Alpha: {lasso_model.alpha_:.4f}")


# --- Ridge (L2) Regularized Regression ---
# Ridge shrinks coefficients but does not set them to zero. Good for multicollinearity.
ridge_model = RidgeCV(cv=5)
ridge_model.fit(X_train, y_train)
ridge_preds = ridge_model.predict(X_test)
ridge_rmse = root_mean_squared_error(y_test, ridge_preds)
print(f"Ridge Regression RMSE: {ridge_rmse:.4f}")
print(f"Ridge Best Alpha: {ridge_model.alpha_:.4f}")


#///////////////////////////////////////////////////////////////////////////////


# --- Decision Tree Regressor/Classifier with Manual Tuning ---

# Split the training data again to create a validation set for tuning
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

depths_to_try = range(1, 11)
validation_scores = []

# Loop through depths to find the best one
for depth in depths_to_try:
    # Use DecisionTreeRegressor for regression, Classifier for classification
    temp_model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    temp_model.fit(X_train_sub, y_train_sub)
    val_preds = temp_model.predict(X_val)
    # For classification, you'd use a metric like accuracy_score
    score = root_mean_squared_error(y_val, val_preds)
    validation_scores.append(score)

# Find the best depth with the lowest error
best_depth = depths_to_try[np.argmin(validation_scores)]
print(f"Best max_depth found: {best_depth}")

# Train the final model on the full training set with the best depth
final_tree_model = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
final_tree_model.fit(X_train, y_train)
tree_test_rmse = root_mean_squared_error(y_test, final_tree_model.predict(X_test))
print(f"Final Decision Tree Test RMSE: {tree_test_rmse:.4f}")


#///////////////////////////////////////////////////////////////////////////////


# --- XGBoost Classifier ---
# NOTE: XGBoost requires labels to be 0 and 1, not -1 and 1.
# First, ensure y is a NumPy array for easier manipulation
y_train_xgb = np.array(y_train)
y_test_xgb = np.array(y_test)

# Convert labels from {-1, 1} or other formats to {0, 1}
# This example assumes the original labels were 1 and -1.
y_train_numeric = np.where(y_train_xgb == 1, 1, 0)
y_test_numeric = np.where(y_test_xgb == 1, 1, 0)

# Initialize and train the XGBoost model
# tree_method='exact' is good for small datasets
xgb_model = xgb.XGBClassifier(tree_method="exact", use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train_numeric)

# Evaluate the model
accuracy = xgb_model.score(X_test, y_test_numeric)
print(f"XGBoost Classifier Accuracy: {accuracy:.2%}")


#///////////////////////////////////////////////////////////////////////////////


# --- AdaBoost from Scratch ---
# This implementation assumes the data has 2 features for plotting.
# The algorithm itself works for any number of features.
# It requires labels to be in {-1, 1} format.

def adaboost_round(X, y, weights):
    """Performs one round of AdaBoost."""
    stump = DecisionTreeClassifier(max_depth=1, random_state=42)
    stump.fit(X, y, sample_weight=weights)
    
    y_pred = stump.predict(X)
    error = np.sum(weights[y_pred != y])
    
    # Add epsilon to prevent division by zero
    epsilon = 1e-10
    model_weight_alpha = 0.5 * np.log((1 - error) / (error + epsilon))
    
    # Update weights
    update_factor = np.exp(-model_weight_alpha * y * y_pred)
    new_weights = weights * update_factor
    new_weights /= np.sum(new_weights) # Normalize
    
    return stump, new_weights, model_weight_alpha

def run_adaboost(X, y, num_rounds):
    """Runs the full AdaBoost algorithm."""
    stumps = []
    alphas = []
    weights = np.ones(len(y)) / len(y) # Initial uniform weights
    
    for _ in range(num_rounds):
        stump, weights, alpha = adaboost_round(X, y, weights)
        stumps.append(stump)
        alphas.append(alpha)
        
    return stumps, alphas

def predict_adaboost(X, stumps, alphas):
    """Makes predictions with the trained AdaBoost ensemble."""
    scores = np.zeros(len(X))
    for stump, alpha in zip(stumps, alphas):
        scores += alpha * stump.predict(X)
    return np.sign(scores)

# --- Example Usage for AdaBoost ---
# Ensure y_train labels are {-1, 1}
y_train_ada = np.where(y_train == 1, 1, -1)
y_test_ada = np.where(y_test == 1, 1, -1)

stumps, alphas = run_adaboost(X_train, y_train_ada, num_rounds=10)
ada_preds = predict_adaboost(X_test, stumps, alphas)
ada_accuracy = np.mean(ada_preds == y_test_ada)
print(f"AdaBoost (from scratch) Accuracy: {ada_accuracy:.2%}")


#///////////////////////////////////////////////////////////////////////////////


# --- Custom MAE function (from HW1) ---
def MAE(true_labels, pred_labels):
    return np.mean(np.abs(np.array(true_labels) - np.array(pred_labels)))

# --- Using scikit-learn metrics ---
# For Regression
mse = mean_squared_error(y_test, lin_preds)
rmse = root_mean_squared_error(y_test, lin_preds)
mae_sklearn = np.mean(np.abs(y_test - lin_preds)) # Or use sklearn.metrics.mean_absolute_error

print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae_sklearn:.4f}")

# For Classification
# 'accuracy' is the fraction of correctly classified samples
# Example using the XGBoost model predictions
accuracy = np.mean(xgb_model.predict(X_test) == y_test_numeric)
print(f"Classification Accuracy: {accuracy:.2%}")


#///////////////////////////////////////////////////////////////////////////////


# --- Plot Classifier Decision Boundary ---
def plot_decision_boundary(clf, X, y, title, save_path=None):
    """
    Plots the decision boundary for a classifier.
    Assumes X has 2 features and y is in {-1, 1} or {0, 1} format.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Use scikit-learn's display function
    DecisionBoundaryDisplay.from_estimator(
        clf, X, response_method="predict",
        ax=ax, alpha=0.3, xlabel="Feature 1", ylabel="Feature 2"
    )
    
    # Scatter plot the data points
    # Ensure y is a NumPy array for boolean indexing
    y_np = np.array(y)
    ax.scatter(X[y_np == 1, 0], X[y_np == 1, 1], marker='+', c='blue', label='Positive Class')
    ax.scatter(X[y_np == 0, 0], X[y_np == 0, 1], marker='_', c='red', label='Negative Class')
    # Use these lines if your labels are {-1, 1}
    # ax.scatter(X[y_np == 1, 0], X[y_np == 1, 1], marker='+', c='blue', label='Positive (+1)')
    # ax.scatter(X[y_np == -1, 0], X[y_np == -1, 1], marker='_', c='red', label='Negative (-1)')

    ax.set_title(title)
    ax.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

# Example usage with the trained XGBoost model (using first 2 features if more exist)
# Note: This requires X_test to be a NumPy array with 2 columns
if X_test.shape[1] >= 2:
    plot_decision_boundary(xgb_model, X_test[:, :2], y_test_numeric, "XGBoost Decision Boundary", "figs/xgboost_boundary.png")


#///////////////////////////////////////////////////////////////////////////////



# --- Save and Load a Model using joblib ---

# 1. Train a model (e.g., our Ridge model)
# ridge_model.fit(X_train, y_train) 

# 2. Save the model to a file
model_filename = 'ridge_regression_model.joblib'
joblib.dump(ridge_model, model_filename)
print(f"Model saved to {model_filename}")

# 3. Load the model from the file in a different session
loaded_model = joblib.load(model_filename)
print("Model loaded successfully.")

# 4. Use the loaded model to make predictions
new_predictions = loaded_model.predict(X_test)
print(f"Predictions from loaded model are the same: {np.allclose(new_predictions, ridge_preds)}")