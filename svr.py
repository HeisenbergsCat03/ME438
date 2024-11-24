import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import mean_squared_error as MSE
from skopt import BayesSearchCV
import matplotlib.pyplot as plt


# Function to preprocess data
def preprocess_data(train_file, test_file, target_col, drop_outliers=False, outlier_condition=None):
    # Load datasets
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    # Handle outliers
    if drop_outliers and outlier_condition:
        train_data = train_data.loc[outlier_condition(train_data)].reset_index(drop=True)
    
    # Split features and targets
    X_train = train_data.iloc[:, 20:]
    y_train = train_data[target_col]
    X_test = test_data.iloc[:, 20:]
    y_test = test_data[target_col]
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test

# Function to train and evaluate SVR
def train_svr(X_train, y_train, X_test, y_test, params):
    # Define model and cross-validation
    model = SVR(kernel='rbf')
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
    
    # Bayesian search for hyperparameter tuning
    search = BayesSearchCV(estimator=model, search_spaces=params, n_jobs=-1, cv=cv)
    search.fit(X_train, y_train)
    print(f"Best Score: {search.best_score_}")
    print(f"Best Parameters: {search.best_params_}")
    
    # Train model with best parameters
    best_model = search.best_estimator_
    best_model.fit(X_train, y_train)
    
    # Evaluate on training and test data
    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)
    train_rmse = np.sqrt(MSE(y_train, train_pred))
    test_rmse = np.sqrt(MSE(y_test, test_pred))
    print(f"Training RMSE: {train_rmse:.3f}, Test RMSE: {test_rmse:.3f}")
    
    return best_model, train_pred, test_pred

# Function to plot predictions
def plot_predictions(y_test, predictions, title, ylabel, save_path):
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(y_test)), y_test, color='red', label='Actual')
    plt.scatter(range(len(predictions)), predictions, color='blue', label='Predicted')
    plt.title(title)
    plt.xlabel('Test Sample Index')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.show()

# Main Script
if __name__ == "__main__":
    # Example: Formation Energy Prediction
    train_file = 'Sine_MP_train.csv'
    test_file = 'sine_mp_test.csv'
    target_col = 'formation_energy_per_atom'
    
    # Define hyperparameter search space
    param_space = {
        'C': (1e-6, 60000.0, 'log-uniform'),
        'gamma': (1e-10, 100.0, 'log-uniform'),
        'epsilon': (1e-10, 10.0, 'log-uniform')
    }
    
    # Preprocess data
    X_train, y_train, X_test, y_test = preprocess_data(
        train_file, test_file, target_col, 
        drop_outliers=True, 
        outlier_condition=lambda df: ~((df['Class_name'] == 'Mgbased') & (df['e_above_hull'] > 0.2))
    )
    
    # Train and evaluate SVR
    model, train_pred, test_pred = train_svr(X_train, y_train, X_test, y_test, param_space)
    
    # Plot results
    plot_predictions(y_test, test_pred, "Formation Energy Prediction", "Energy (eV/atom)", "formation_energy.png")
