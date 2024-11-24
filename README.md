# Silicon-Metal Alloy Potentials Project

## Overview
This project investigates the use of machine learning (ML) techniques to model and predict interatomic potentials in silicon-metal alloy systems. It combines Density Functional Theory (DFT) calculations with ML to achieve accurate and computationally efficient potential predictions. Three ML models—Support Vector Regression (SVR), Gaussian Mixture Models (GMM), and Fully Connected Neural Networks (FCNN)—are trained on the generated datasets.

## Workflow

### 1. Install Requirements
Install the required Python libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## 2. Data Collection and Preprocessing
Run `data_collection.ipynb` to:
- **Collect material properties** using libraries like `pymatgen` and `matminer`.
- **Perform DFT calculations** to generate potential energy surface data.
- **Process and structure data** into CSV files for training.

### Key Datasets Generated:
- **Material Properties**: Includes energy per atom, formation energy per atom, band gap, etc.
- **Categorical Data**: Material classifications and labels.
- **Featurized Data**: Includes density features, XRD powder patterns, orbital field matrices, DFT-based generated data.

---

## 3. Model Training
Train three ML models on the generated datasets:
- **`nn.py`**: Trains a Fully Connected Neural Network (FCNN) for predicting energy-related properties.
- **`svr.py`**: Trains a Support Vector Regression (SVR) model to predict formation and potential energies.
- **`gmm.py`**: Fits a Gaussian Mixture Model (GMM) to probabilistically model energy distributions.

### Each Script:
- Outputs **model performance metrics** (e.g., RMSE).

---

## 4. Plot Results
Run `plot.ipynb` to visualize:
- **Actual vs. Predicted Potentials** for each ML model.
- **RMSE performance** across models and datasets.
- **Comparisons** of training and testing RMSE for each technique.

The notebook reproduces the figures shown in the project report.

---

## Results

### Performance Metrics
The models were evaluated using **Root Mean Square Error (RMSE)**:
- **Support Vector Regression (SVR)**: Achieved the lowest RMSE, showing strong predictive performance and good generalization.
- **Gaussian Mixture Models (GMM)**: Moderate RMSE values but struggled with generalization.
- **Neural Networks (NN)**: Highest RMSE, indicating overfitting and poor generalization.

### Dataset Observations
- **XRD Dataset**: Best performance for all models, particularly SVR.
- **Orbital and Sine Datasets**: SVR still outperformed other models, but with slightly higher RMSE.
- **DFT Dataset**: Most challenging for all models, with NN showing the poorest performance.

### Comparative Metrics
| Dataset | Model | Train RMSE | Test RMSE |
|---------|-------|------------|-----------|
| XRD     | SVR   | 0.095      | 0.087     |
| XRD     | GMM   | 0.476      | 1.323     |
| XRD     | NN    | 0.731      | 1.874     |

---

## Acknowledgment
This project was developed as part of the ME438 course at IIT Bombay, under the guidance of **Prof. Amit Singh**. 

