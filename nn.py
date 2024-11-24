import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and Preprocess Data
def preprocess_data(train_file, test_file, target_col):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Features and Target
    X_train = train_data.iloc[:, 14:]
    y_train = train_data[target_col]
    X_test = test_data.iloc[:, 14:]
    y_test = test_data[target_col]

    # Normalize Features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    return X_train, y_train, X_test, y_test

# Define the FCNN Model
class FCNN(nn.Module):
    def __init__(self, input_dim):
        super(FCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

# Train the Model
def train_model(model, criterion, optimizer, X_train, y_train, X_test, y_test, epochs=100):
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        train_pred = model(X_train)
        train_loss = criterion(train_pred, y_train)
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test)

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

    return train_losses, test_losses

# Plot Results
def plot_results(y_test, predictions, title, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
    plt.scatter(range(len(predictions)), predictions, color='red', label='Predicted')
    plt.title(title)
    plt.xlabel('Test Sample Index')
    plt.ylabel('Energy per Atom')
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.show()

# Main Execution
if __name__ == "__main__":
    train_file = 'Xrd_MP_train.csv'
    test_file = 'xrd_test_mp.csv'
    target_col = 'energy_per_atom'

    # Preprocess data
    X_train, y_train, X_test, y_test = preprocess_data(train_file, test_file, target_col)

    # Define model, loss, and optimizer
    input_dim = X_train.shape[1]
    model = FCNN(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_losses, test_losses = train_model(model, criterion, optimizer, X_train, y_train, X_test, y_test, epochs=20)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).numpy()
        y_test_np = y_test.numpy()
        rmse = np.sqrt(MSE(y_test_np, test_pred))
        print(f"Test RMSE: {rmse:.3f}")

    # Plot predictions
    plot_results(y_test_np.flatten(), test_pred.flatten(), "FCNN: Actual vs Predicted", "fcnn_actual_vs_predicted.png")
