import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

train_file = 'Sine_MP_train.csv'
test_file = 'sine_mp_test.csv'

def preprocess_data(train_file, test_file, target_col):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    X_train = train_data.iloc[:, 20:]
    y_train = train_data[target_col]
    X_test = test_data.iloc[:, 20:]
    y_test = test_data[target_col]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test

def gmm_regression(X_train, y_train, X_test, y_test, n_components=5):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_train)

    # Assign cluster means as predictions
    cluster_means = np.array([y_train[gmm.predict(X_train) == i].mean() for i in range(n_components)])
    train_pred = cluster_means[gmm.predict(X_train)]
    test_pred = cluster_means[gmm.predict(X_test)]

    train_rmse = np.sqrt(MSE(y_train, train_pred))
    test_rmse = np.sqrt(MSE(y_test, test_pred))
    print(f"GMM - Train RMSE: {train_rmse:.3f}, Test RMSE: {test_rmse:.3f}")

    return train_pred, test_pred

# Plot Results
def plot_results(y_test, predictions, title, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
    plt.scatter(range(len(predictions)), predictions, color='red', label='Predicted')
    plt.title(title)
    plt.xlabel('Test Sample Index')
    plt.ylabel('Energy per Atom')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.show()

# Main Execution
if __name__ == "__main__":
    target_col = 'energy_per_atom'
    X_train, y_train, X_test, y_test = preprocess_data(train_file, test_file, target_col)
    train_pred, test_pred = gmm_regression(X_train, y_train, X_test, y_test, n_components=5)
    plot_results(y_test, test_pred, "GMM: Actual vs Predicted", "gmm_actual_vs_predicted.png")
