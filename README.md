# README: SSE-BGP: Feature Engineering, Autoencoder, and Stacked Regressor Pipeline

This project demonstrates a comprehensive workflow for feature encoding, dimensionality reduction using an autoencoder, and model stacking with various regression models. The final ensemble uses a PyTorch-based meta-learner for optimal predictions.

---

## Overview

The pipeline is divided into two phases:

1. **Feature Engineering and Autoencoder**:
   - Preprocess and encode features.
   - Dimensionality reduction via an advanced autoencoder model.

2. **Stacked Regression Model**:
   - Use various base regressors (Random Forest, XGBoost, CatBoost, Gradient Boosting, LightGBM).
   - Combine them using a PyTorch-based meta-learner in a stacking regressor for robust predictions.

The goal is to predict the target variable (`HSE06_Band_Gap`) accurately using a combination of advanced feature extraction and ensemble modeling.

---

## Requirements

### Python Libraries

Install the following libraries to run the project:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost catboost lightgbm torch
```

---

## Pipeline Phases

### Phase 1: Feature Engineering and Autoencoder

#### Preprocessing
1. Load and preprocess data:
   - Encode categorical variables using one-hot encoding.
   - Normalize numerical variables.
2. Dimensionality reduction:
   - Use an autoencoder to reduce features to a smaller representation.

#### Autoencoder Details
- **Architecture**:
  - Encoder: Multi-layer neural network with dropout and batch normalization.
  - Decoder: Symmetric structure to reconstruct input features.
- **Training**:
  - Loss: Mean Squared Error (MSE).
  - Optimizer: Adam with a learning rate of 0.0001.
  - Early stopping to prevent overfitting.

#### Outputs
- Encoded features saved as:
  - `Encoded_train_data.csv`
  - `Encoded_test_data.csv`
- Target variables saved as:
  - `y_train.csv`
  - `y_test.csv`

---

### Phase 2: Stacked Regression Model

#### Base Regressors
- Random Forest (`RandomForestRegressor`)
- XGBoost (`XGBRegressor`)
- CatBoost (`CatBoostRegressor`)
- Gradient Boosting Regressor (`GradientBoostingRegressor`)
- LightGBM (`LGBMRegressor`)

#### Meta-Learner
- **PyTorch-based Regressor**:
  - Fully connected neural network with two hidden layers, ReLU activation, and dropout.
  - Optimized using Adam.

#### Stacking Implementation
1. Combine base learners' predictions as features for the meta-learner.
2. Meta-learner uses a 5-fold cross-validation approach for robust predictions.

---

## Model Performance

### Metrics
The following metrics are computed to evaluate the stacked regressor:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² (coefficient of determination)

### Residual Analysis
- Distribution of residuals is visualized with a histogram and kernel density estimation (KDE).
- Key insights:
  - Mean of residuals.
  - Standard deviation of residuals.

---

## Execution Steps

### Step 1: Preprocessing and Autoencoder
1. Run the autoencoder pipeline to generate encoded feature files (`Encoded_train_data.csv` and `Encoded_test_data.csv`).
2. Save target variables in corresponding `.csv` files.

### Step 2: Load Encoded Features and Targets
1. Load preprocessed data files:
   - `Encoded_train_data.csv`
   - `Encoded_test_data.csv`
   - `y_train.csv`
   - `y_test.csv`

### Step 3: Train Stacked Regressor
1. Define base learners and meta-learner.
2. Train the stacked regressor using:
   ```python
   stacking_regressor.fit(X_train_encoded, y_train)
   ```

### Step 4: Evaluate and Analyze Results
1. Predict on the test set:
   ```python
   y_pred_stack = stacking_regressor.predict(X_test_encoded)
   ```
2. Compute evaluation metrics:
   ```python
   mse = mean_squared_error(y_test, y_pred_stack)
   r2 = r2_score(y_test, y_pred_stack)
   ```
3. Analyze residuals:
   - Plot the distribution of residuals.
   - Annotate key statistics.

---

## Outputs

### Key Results
- Performance metrics (`MSE`, `RMSE`, `R²`, `MAE`) printed in the console.
- Residuals distribution visualized and annotated.
- Encoded features (`Encoded_train_data.csv`, `Encoded_test_data.csv`).

### Visualization
- **Correlation Heatmap**: Shows relationships among features.
- **Residuals Plot**: Assesses error distribution.

---

## Customization

### Model Parameters
- Adjust autoencoder dimensions, learning rates, and dropout rates.
- Fine-tune base learners using hyperparameter optimization.

### Feature Selection
- Modify the feature correlation threshold to explore different subsets of data.
- Use PCA or SelectKBest for alternative dimensionality reduction methods.

---

## Database

In this study, we selected 2D material data from the Computational 2D Materials Database (C2DB), a high-quality publicly available resource widely used in materials science research, as the source of input features for the model. The C2DB database is a high-quality publicly available resource widely used in materials science research, focusing on computational structures and properties of 2D materials. Its data are generated through DFT-based high-throughput calculations covering a large collection of 2D materials, including their electronic, optical, and mechanical properties. This makes it an ideal basis for studying tasks such as band gap prediction.

Haastrup, S., Strange, M., Pandey, M., Deilmann, T., Schmidt, P. S., Hinsche, N. F., ... & Thygesen, K. S. (2018).
"The Computational 2D Materials Database: High-Throughput Modeling and Discovery of Atomically Thin Crystals."
2D Materials, 5(4), 042002.
DOI: 10.1088/2053-1583/aacfc1

Gjerding, M. N., Taghizadeh, A., Rasmussen, A., Ali, S., Bertoldo, F., Deilmann, T., ... & Thygesen, K. S. (2021).
"Recent Progress of the Computational 2D Materials Database (C2DB)."
2D Materials, 8(4), 044002.
DOI: 10.1088/2053-1583/abf15d

---

For questions or further customization, feel free to reach out!
